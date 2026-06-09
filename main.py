import cv2
import mediapipe as mp
import numpy as np
import math
import time
from datetime import datetime
import os
from image_to_text import image_to_text
from preprocess import preprocess_image
import speech_recognition as sr
import threading


class OneEuroFilter:
    """Adaptive low-pass filter for noisy pointing signals (Casiez et al. 2012).

    A fixed smoothing factor forces a trade-off: smooth but laggy, or
    responsive but jittery. The One Euro filter instead adapts its cutoff to
    the signal speed. When the fingertip is nearly still it filters hard to
    remove tracking jitter; when it moves quickly it loosens up so the cursor
    keeps pace with the hand. This makes handwriting strokes both steady and
    responsive, which in turn yields cleaner lines for OCR.
    """

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        # min_cutoff: lower = more smoothing of slow movements (less jitter).
        # beta:       higher = less lag during fast movements.
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        """Forget history so the next sample is taken as-is (avoids a jump
        when the hand reappears in a new spot)."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x, timestamp=None):
        if timestamp is None:
            timestamp = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = timestamp
            return x

        dt = timestamp - self.t_prev
        if dt <= 0:
            dt = 1e-3  # guard against duplicate/zero timestamps
        self.t_prev = timestamp

        # Filter the derivative, then use it to drive the adaptive cutoff.
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

class TouchlessWritingSystem:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Canvas settings
        self.canvas_width = 1280
        self.canvas_height = 720
        self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        
        # Webcam settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Drawing settings
        self.colors = [
            ('Black', (0, 0, 0)),
            ('Red', (0, 0, 255)),
            ('Blue', (255, 0, 0)),
            ('Green', (0, 255, 0)),
            ('Purple', (128, 0, 128)),
            ('Orange', (0, 165, 255)),
            ('Yellow', (0, 255, 255)),
            ('Pink', (203, 192, 255))
        ]
        self.current_color_idx = 0
        self.current_color = self.colors[0][1]
        self.brush_size = 3
        self.eraser_size = 30
        
        # State management
        self.drawing_mode = False
        self.erasing_mode = False
        self.last_position = None
        self.current_gesture = 'idle'
        self.gesture_stability = {'gesture': 'idle', 'count': 0, 'threshold': 3}
        
        # Cursor smoothing — adaptive One Euro filter on each axis. Beats a
        # fixed EMA: steady when the fingertip hovers, snappy when it darts.
        self.filter_x = OneEuroFilter(min_cutoff=1.0, beta=0.02)
        self.filter_y = OneEuroFilter(min_cutoff=1.0, beta=0.02)
        self.smoothed_position = None
        
        # Color selection state
        self.selection_timer = 0
        self.selection_threshold = 15  # Frames to hold before selecting
        self.last_hovered_color = None
        
        # Multi-page support
        self.pages = [np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255]
        self.current_page = 0
        
        # Drawing history for undo
        self.drawing_points = []
        
        # UI elements
        self.setup_ui()
        
        # Status message
        self.status_message = "Ready! Show hand gestures to start"
        self.message_timer = 0
        
        # Live text conversion mode
        self.live_text_mode = False
        self.extracted_text = ""
        self.text_scroll_offset = 0

        # OCR preprocessing pipeline toggle (press X to flip)
        self.use_preprocessing = True
        
        # Voice to text mode
        self.voice_mode = False
        self.is_listening = False
        self.voice_text = ""
        self.recognizer = sr.Recognizer()
        self.voice_thread = None
        
    def setup_ui(self):
        """Setup UI elements like color palette"""
        self.color_palette_x = 20
        self.color_palette_y = 80
        self.color_box_width = 60
        self.color_box_height = 50
        self.color_box_spacing = 10
        
    @staticmethod
    def _distance(a, b):
        """Euclidean distance between two MediaPipe landmarks (uses z too)."""
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    def _palm_size(self, hand_landmarks):
        """Reference hand scale: wrist to middle-finger MCP.

        Used to normalize thresholds so detection works whether the hand is
        near or far from the camera.
        """
        lm = hand_landmarks.landmark
        return self._distance(lm[0], lm[9]) + 1e-6

    def detect_finger_state(self, hand_landmarks, finger_tip_id, finger_pip_id):
        """Check if a finger is extended (rotation-invariant).

        A finger is extended when its tip is farther from the wrist than its
        PIP joint AND the finger is reasonably straight. This holds in any hand
        orientation, unlike a raw y-axis comparison which only works for an
        upright hand. The straightness test (tip-to-MCP span vs. the summed
        bone lengths) rejects a curled finger whose tip happens to sit far from
        the wrist, cutting down on false "extended" readings while pointing.
        """
        lm = hand_landmarks.landmark
        wrist = lm[0]
        mcp = lm[finger_pip_id - 1]   # MCP is one joint below the PIP
        pip = lm[finger_pip_id]
        dip = lm[finger_tip_id - 1]   # DIP is one joint below the tip
        tip = lm[finger_tip_id]

        # Normalize the margin by hand scale so it adapts to camera distance.
        margin = 0.05 * self._palm_size(hand_landmarks)
        farther_than_pip = self._distance(tip, wrist) > self._distance(pip, wrist) + margin

        # Straightness: a straight finger's tip-to-MCP span is close to the
        # sum of its bone lengths; a curled one's span is much shorter.
        bone_len = (self._distance(mcp, pip)
                    + self._distance(pip, dip)
                    + self._distance(dip, tip))
        straightness = self._distance(tip, mcp) / (bone_len + 1e-6)

        return farther_than_pip and straightness > 0.7

    def detect_thumb_state(self, hand_landmarks):
        """Check if the thumb is extended (rotation- and handedness-invariant).

        The thumb is extended when its tip sits farther from the pinky's base
        than its IP joint does. This avoids the brittle left/right x-axis
        assumption of a raw coordinate comparison.
        """
        lm = hand_landmarks.landmark
        pinky_mcp = lm[17]
        thumb_tip = lm[4]
        thumb_ip = lm[3]

        margin = 0.05 * self._palm_size(hand_landmarks)
        return self._distance(thumb_tip, pinky_mcp) > self._distance(thumb_ip, pinky_mcp) + margin
    
    def detect_gesture(self, hand_landmarks):
        """Detect hand gesture based on finger positions"""
        # Check each finger
        is_thumb = self.detect_thumb_state(hand_landmarks)
        is_index = self.detect_finger_state(hand_landmarks, 8, 6)
        is_middle = self.detect_finger_state(hand_landmarks, 12, 10)
        is_ring = self.detect_finger_state(hand_landmarks, 16, 14)
        is_pinky = self.detect_finger_state(hand_landmarks, 20, 18)
        
        # Gesture detection logic
        # Index only = Move cursor
        if is_index and not is_middle and not is_ring and not is_pinky and not is_thumb:
            return 'move'
        
        # Index + Thumb = Draw
        if is_index and is_thumb and not is_middle and not is_ring and not is_pinky:
            return 'draw'
        
        # Index + Middle = Erase
        if is_index and is_middle and not is_ring and not is_pinky and not is_thumb:
            return 'erase'
        
        # Index + Pinky = Select
        if is_index and is_pinky and not is_middle and not is_ring:
            return 'select'
        
        # All fingers = Clear page
        if is_thumb and is_index and is_middle and is_ring and is_pinky:
            return 'clear'
        
        return 'idle'
    
    def stabilize_gesture(self, gesture):
        """Stabilize gesture detection to avoid flickering"""
        if gesture == self.gesture_stability['gesture']:
            self.gesture_stability['count'] += 1
        else:
            self.gesture_stability['gesture'] = gesture
            self.gesture_stability['count'] = 1
        
        if self.gesture_stability['count'] >= self.gesture_stability['threshold']:
            return gesture
        
        return self.current_gesture
    
    def get_finger_position(self, hand_landmarks, frame_shape):
        """Get index finger tip position with margin for edge detection"""
        index_tip = hand_landmarks.landmark[8]
        h, w, _ = frame_shape
        
        # Add margin to allow detection at edges
        # Map from [-margin, 1+margin] range to [0, canvas_width/height]
        margin = -0.15  # 15% margin on each side
        
        # Normalize coordinates with margin
        normalized_x = (index_tip.x + margin) / (1 + 2 * margin)
        normalized_y = (index_tip.y + margin) / (1 + 2 * margin)
        
        # Map to canvas coordinates
        x = int(normalized_x * self.canvas_width)
        y = int(normalized_y * self.canvas_height)
        
        # Clamp to canvas boundaries
        x = max(0, min(self.canvas_width - 1, x))
        y = max(0, min(self.canvas_height - 1, y))
        
        return (x, y)
    
    def smooth_position(self, new_position):
        """Smooth the fingertip position with a per-axis One Euro filter.

        Returns a steady cursor when the finger hovers and a responsive one
        when it moves quickly, giving cleaner handwriting strokes.
        """
        new_x, new_y = new_position
        now = time.time()

        smooth_x = int(round(self.filter_x.filter(new_x, now)))
        smooth_y = int(round(self.filter_y.filter(new_y, now)))

        self.smoothed_position = (smooth_x, smooth_y)
        return self.smoothed_position

    def reset_smoothing(self):
        """Clear filter history so the cursor doesn't jump when the hand
        reappears after being lost."""
        self.filter_x.reset()
        self.filter_y.reset()
        self.smoothed_position = None
    
    def draw_color_palette(self, display_canvas):
        """Draw color selection palette with hover feedback"""
        for idx, (name, color) in enumerate(self.colors):
            x = self.color_palette_x
            y = self.color_palette_y + idx * (self.color_box_height + self.color_box_spacing)
            
            # Draw color box
            cv2.rectangle(display_canvas, 
                         (x, y), 
                         (x + self.color_box_width, y + self.color_box_height),
                         color, -1)
            
            # Draw border (thicker for selected color, highlighted for hovered)
            if idx == self.current_color_idx:
                border_thickness = 4
                border_color = (255, 255, 255)
            elif idx == self.last_hovered_color and self.current_gesture == 'select':
                # Show hover feedback
                border_thickness = 3
                border_color = (0, 255, 255)  # Cyan for hover
            else:
                border_thickness = 2
                border_color = (100, 100, 100)
                
            cv2.rectangle(display_canvas, 
                         (x, y), 
                         (x + self.color_box_width, y + self.color_box_height),
                         border_color, border_thickness)
            
            # Draw selection progress bar if hovering
            if idx == self.last_hovered_color and self.selection_timer > 0:
                progress = min(1.0, self.selection_timer / self.selection_threshold)
                bar_width = int(self.color_box_width * progress)
                cv2.rectangle(display_canvas,
                            (x, y + self.color_box_height + 2),
                            (x + bar_width, y + self.color_box_height + 6),
                            (0, 255, 255), -1)
            
            # Draw color name
            cv2.putText(display_canvas, name, 
                       (x + self.color_box_width + 10, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def check_color_selection(self, x, y):
        """Check if cursor is over a color box and return the color index"""
        for idx in range(len(self.colors)):
            box_x = self.color_palette_x
            box_y = self.color_palette_y + idx * (self.color_box_height + self.color_box_spacing)
            
            if (box_x <= x <= box_x + self.color_box_width and 
                box_y <= y <= box_y + self.color_box_height):
                return idx
        return None
    
    def select_color(self, color_idx):
        """Select a color by index"""
        if color_idx is not None and 0 <= color_idx < len(self.colors):
            self.current_color_idx = color_idx
            self.current_color = self.colors[color_idx][1]
            self.set_status(f"Color changed to {self.colors[color_idx][0]}")
            return True
        return False
    
    def handle_gesture(self, gesture, position):
        """Handle different gestures"""
        x, y = position
        
        if gesture == 'move':
            self.drawing_mode = False
            self.erasing_mode = False
            self.last_position = position
            self.selection_timer = 0  # Reset selection timer
            self.last_hovered_color = None
            
        elif gesture == 'draw':
            if not self.drawing_mode:
                self.drawing_mode = True
                self.last_position = position
                self.drawing_points = [position]
            else:
                if self.last_position:
                    cv2.line(self.canvas, self.last_position, position,
                            self.current_color, self.brush_size, lineType=cv2.LINE_AA)
                    self.drawing_points.append(position)
                self.last_position = position
            self.selection_timer = 0  # Reset selection timer
                
        elif gesture == 'erase':
            self.erasing_mode = True
            self.drawing_mode = False
            cv2.circle(self.canvas, position, self.eraser_size, (255, 255, 255), -1)
            self.selection_timer = 0  # Reset selection timer
            
        elif gesture == 'select':
            self.drawing_mode = False
            self.erasing_mode = False
            
            # Check which color box is being hovered
            hovered_color_idx = self.check_color_selection(x, y)
            
            if hovered_color_idx is not None:
                # If hovering over the same color, increment timer
                if hovered_color_idx == self.last_hovered_color:
                    self.selection_timer += 1
                    
                    # Select color after holding for threshold frames
                    if self.selection_timer >= self.selection_threshold:
                        self.select_color(hovered_color_idx)
                        self.selection_timer = 0  # Reset after selection
                else:
                    # New color being hovered, reset timer
                    self.last_hovered_color = hovered_color_idx
                    self.selection_timer = 1
            else:
                # Not hovering over any color, reset
                self.selection_timer = 0
                self.last_hovered_color = None
            
        elif gesture == 'clear':
            self.clear_page()
            self.selection_timer = 0  # Reset selection timer
            
        else:  # idle
            self.drawing_mode = False
            self.erasing_mode = False
            self.selection_timer = 0  # Reset selection timer
            self.last_hovered_color = None
    
    def draw_cursor(self, canvas, position, gesture):
        """Draw custom cursor based on gesture"""
        x, y = position
        
        if gesture == 'move':
            cv2.circle(canvas, position, 10, (255, 200, 0), 2)
        elif gesture == 'draw':
            cv2.circle(canvas, position, 10, (0, 255, 0), -1)
            cv2.circle(canvas, position, 12, (255, 255, 255), 2)
        elif gesture == 'erase':
            cv2.circle(canvas, position, self.eraser_size, (255, 0, 0), 2)
        elif gesture == 'select':
            cv2.circle(canvas, position, 15, (255, 0, 255), 2)
        else:
            cv2.circle(canvas, position, 8, (200, 200, 200), 2)
    
    def draw_status_bar(self, display_canvas):
        """Draw status bar with information"""
        # Background
        cv2.rectangle(display_canvas, (0, 0), (self.canvas_width, 60), (50, 50, 50), -1)
        
        # Title
        cv2.putText(display_canvas, "Touchless Writing System", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Page info
        page_text = f"Page {self.current_page + 1}/{len(self.pages)}"
        cv2.putText(display_canvas, page_text, 
                   (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Gesture status
        gesture_colors = {
            'move': (0, 255, 255),
            'draw': (0, 255, 0),
            'erase': (0, 0, 255),
            'select': (255, 0, 255),
            'idle': (150, 150, 150)
        }
        gesture_color = gesture_colors.get(self.current_gesture, (150, 150, 150))
        
        cv2.putText(display_canvas, f"Status: {self.status_message}", 
                   (700, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 1)
    
    def draw_instructions(self, display_canvas):
        """Draw instruction panel"""
        instructions = [
            "GESTURES:",
            "Move: Index only",
            "Draw: Index + Thumb",
            "Erase: Index + Middle",
            "Select: Index + Pinky",
            "Clear: All fingers",
            "",
            "KEYS:",
            "N: New page",
            "P: Previous page",
            "S: Save image",
            "T: Text to file",
            "L: Live text mode",
            "V: Voice to text",
            "C: Clear page",
            "Q: Quit"
        ]
        
        y_offset = self.canvas_height - 450
        for i, text in enumerate(instructions):
            cv2.putText(display_canvas, text,
                       (self.canvas_width - 250, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def set_status(self, message):
        """Set status message"""
        self.status_message = message
        self.message_timer = 60  # Display for 60 frames
    
    def clear_page(self):
        """Clear current page"""
        self.pages[self.current_page] = np.ones(
            (self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        self.canvas = self.pages[self.current_page].copy()
        self.set_status("Page cleared")
    
    def new_page(self):
        """Create a new page"""
        new_canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        self.pages.append(new_canvas)
        self.current_page = len(self.pages) - 1
        self.canvas = self.pages[self.current_page].copy()
        self.set_status(f"New page created - Page {self.current_page + 1}")
    
    def next_page(self):
        """Go to next page"""
        if self.current_page < len(self.pages) - 1:
            self.pages[self.current_page] = self.canvas.copy()
            self.current_page += 1
            self.canvas = self.pages[self.current_page].copy()
            self.set_status(f"Page {self.current_page + 1}/{len(self.pages)}")
    
    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.pages[self.current_page] = self.canvas.copy()
            self.current_page -= 1
            self.canvas = self.pages[self.current_page].copy()
            self.set_status(f"Page {self.current_page + 1}/{len(self.pages)}")
    
    def save_image(self):
        """Save current page as image"""
        if not os.path.exists('saved_notes'):
            os.makedirs('saved_notes')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_notes/page_{self.current_page + 1}_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        self.set_status(f"Saved: {filename}")
    
    def save_all_pages(self):
        """Save all pages"""
        if not os.path.exists('saved_notes'):
            os.makedirs('saved_notes')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, page in enumerate(self.pages):
            filename = f"saved_notes/all_pages_{timestamp}_page_{idx + 1}.png"
            cv2.imwrite(filename, page)
        
        self.set_status(f"Saved {len(self.pages)} pages")
    
    def convert_page_to_text(self):
        """Convert current page to text using OCR"""
        try:
            # Create temp directory if it doesn't exist
            if not os.path.exists('temp'):
                os.makedirs('temp')
            
            # Save current page as temporary image
            temp_filename = 'temp/current_page_temp.png'
            cv2.imwrite(temp_filename, self.canvas)
            ocr_input = preprocess_image(temp_filename, out_path='temp/current_page_temp_pre.png', enable=self.use_preprocessing)

            # Use image_to_text function to extract text
            self.set_status("Converting image to text...")
            extracted_text = image_to_text(ocr_input)
            
            # Save extracted text to file
            if not os.path.exists('extracted_text'):
                os.makedirs('extracted_text')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_filename = f"extracted_text/page_{self.current_page + 1}_{timestamp}.txt"
            
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(f"Extracted Text from Page {self.current_page + 1}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(extracted_text)
            
            # Also print to console
            print("\n" + "="*60)
            print(f"TEXT EXTRACTED FROM PAGE {self.current_page + 1}")
            print("="*60)
            print(extracted_text)
            print("="*60)
            print(f"Saved to: {text_filename}\n")
            
            self.set_status(f"Text extracted and saved to {text_filename}")
            
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            return extracted_text
            
        except Exception as e:
            error_msg = f"Error converting to text: {str(e)}"
            print(error_msg)
            self.set_status(error_msg)
            return None
    
    def toggle_live_text_mode(self):
        """Toggle live text conversion mode"""
        if not self.live_text_mode:
            # Entering live text mode - convert current page
            self.set_status("Converting to text... Please wait")
            
            try:
                # Create temp directory if it doesn't exist
                if not os.path.exists('temp'):
                    os.makedirs('temp')
                
                # Save current page as temporary image
                temp_filename = 'temp/live_convert_temp.png'
                cv2.imwrite(temp_filename, self.canvas)
                ocr_input = preprocess_image(temp_filename, out_path='temp/live_convert_temp_pre.png', enable=self.use_preprocessing)

                # Use image_to_text function to extract text
                self.extracted_text = image_to_text(ocr_input)
                
                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                # Enable live text mode
                self.live_text_mode = True
                self.text_scroll_offset = 0
                self.set_status("Live Text Mode - Press L to exit | ↑↓ to scroll")
                
                print("\n" + "="*60)
                print("LIVE TEXT MODE ACTIVATED")
                print("="*60)
                print(self.extracted_text)
                print("="*60 + "\n")
                
            except Exception as e:
                error_msg = f"Error in live conversion: {str(e)}"
                print(error_msg)
                self.set_status(error_msg)
                self.live_text_mode = False
        else:
            # Exiting live text mode
            self.live_text_mode = False
            self.extracted_text = ""
            self.text_scroll_offset = 0
            self.set_status("Live Text Mode disabled")
            print("Live Text Mode disabled\n")
    
    def draw_live_text_overlay(self, display_canvas):
        """Draw extracted text overlay on the canvas"""
        if not self.live_text_mode or not self.extracted_text:
            return
        
        # Create semi-transparent overlay
        overlay = display_canvas.copy()
        
        # Draw background rectangle
        margin = 100
        bg_x1 = margin
        bg_y1 = 100
        bg_x2 = self.canvas_width - margin
        bg_y2 = self.canvas_height - 100
        
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (40, 40, 40), -1)
        
        # Apply transparency
        alpha = 0.92
        cv2.addWeighted(overlay, alpha, display_canvas, 1 - alpha, 0, display_canvas)
        
        # Draw border
        cv2.rectangle(display_canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 255), 3)
        
        # Draw title
        cv2.putText(display_canvas, "EXTRACTED TEXT (Live Mode)", 
                   (bg_x1 + 20, bg_y1 + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw close instruction
        cv2.putText(display_canvas, "Press L to exit | Up/Down arrows to scroll", 
                   (bg_x1 + 20, bg_y1 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw separator line
        cv2.line(display_canvas, 
                (bg_x1 + 10, bg_y1 + 70), 
                (bg_x2 - 10, bg_y1 + 70),
                (100, 100, 100), 2)
        
        # Prepare text for display
        font_scale = 0.5
        font_thickness = 1
        line_height = 25
        max_width = bg_x2 - bg_x1 - 40
        
        # Word wrap the text
        words = self.extracted_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 
                                                  font_scale, font_thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Calculate visible area
        text_area_height = bg_y2 - bg_y1 - 100
        max_visible_lines = int(text_area_height / line_height)
        
        # Apply scroll offset
        start_line = self.text_scroll_offset
        end_line = min(start_line + max_visible_lines, len(lines))
        
        # Draw text lines
        y_pos = bg_y1 + 100
        for i in range(start_line, end_line):
            cv2.putText(display_canvas, lines[i],
                       (bg_x1 + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            y_pos += line_height
        
        # Draw scroll indicator if needed
        if len(lines) > max_visible_lines:
            scroll_percentage = start_line / max(1, len(lines) - max_visible_lines)
            indicator_height = max(20, int((max_visible_lines / len(lines)) * text_area_height))
            indicator_y = bg_y1 + 100 + int(scroll_percentage * (text_area_height - indicator_height))
            
            cv2.rectangle(display_canvas,
                         (bg_x2 - 25, indicator_y),
                         (bg_x2 - 15, indicator_y + indicator_height),
                         (0, 255, 255), -1)
            
            # Show line numbers
            cv2.putText(display_canvas, f"{start_line + 1}-{end_line}/{len(lines)}",
                       (bg_x2 - 80, bg_y2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def scroll_text(self, direction):
        """Scroll the live text display"""
        if not self.live_text_mode:
            return
        
        # Calculate max lines
        words = self.extracted_text.split()
        lines = []
        current_line = ""
        max_width = self.canvas_width - 240
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.5, 1)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        total_lines = len(lines)
        text_area_height = self.canvas_height - 300
        max_visible_lines = int(text_area_height / 25)
        max_scroll = max(0, total_lines - max_visible_lines)
        
        # Update scroll offset
        if direction == 'up':
            self.text_scroll_offset = max(0, self.text_scroll_offset - 1)
        elif direction == 'down':
            self.text_scroll_offset = min(max_scroll, self.text_scroll_offset + 1)
    
    def listen_for_voice(self):
        """Background thread to listen for voice input"""
        try:
            with sr.Microphone() as source:
                self.set_status("🎤 Listening... Speak now!")
                print("\n🎤 Listening for voice input...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                self.is_listening = True
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                self.is_listening = False
                
                self.set_status("Processing speech...")
                print("Processing speech...")
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                
                self.voice_text = text
                self.set_status(f"✅ Recognized: {text}")
                print(f"✅ Recognized: {text}")
                
                # Write the text on canvas
                self.write_text_on_canvas(text)
                
        except sr.WaitTimeoutError:
            self.set_status("⏱️ Timeout - No speech detected")
            print("⏱️ Timeout - No speech detected")
            self.is_listening = False
        except sr.UnknownValueError:
            self.set_status("❌ Could not understand audio")
            print("❌ Could not understand audio")
            self.is_listening = False
        except sr.RequestError as e:
            self.set_status(f"❌ Error: {e}")
            print(f"❌ Speech recognition error: {e}")
            self.is_listening = False
        except Exception as e:
            self.set_status(f"❌ Error: {str(e)}")
            print(f"❌ Voice error: {str(e)}")
            self.is_listening = False
    
    def toggle_voice_mode(self):
        """Toggle voice to text mode"""
        if not self.voice_mode:
            self.voice_mode = True
            self.set_status("🎤 Voice Mode Active - Press V to start listening, ESC to exit")
            print("\n" + "="*60)
            print("🎤 VOICE MODE ACTIVATED")
            print("="*60)
            print("Press 'V' to start voice recording")
            print("Press 'ESC' to exit voice mode")
            print("="*60 + "\n")
        else:
            self.voice_mode = False
            self.set_status("Voice mode disabled")
            print("Voice mode disabled\n")
    
    def start_voice_recording(self):
        """Start voice recording in a separate thread"""
        if self.is_listening:
            self.set_status("⚠️ Already listening...")
            return
        
        if self.voice_thread and self.voice_thread.is_alive():
            return
        
        # Start listening in background thread
        self.voice_thread = threading.Thread(target=self.listen_for_voice)
        self.voice_thread.daemon = True
        self.voice_thread.start()
    
    def write_text_on_canvas(self, text):
        """Write recognized text on the canvas"""
        if not text:
            return
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        line_height = 40
        margin = 50
        max_width = self.canvas_width - 2 * margin
        
        # Split text into words and create lines that fit
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Find a suitable position to write (center of canvas)
        start_y = (self.canvas_height - len(lines) * line_height) // 2
        
        # Draw background rectangle for text
        total_height = len(lines) * line_height + 40
        cv2.rectangle(self.canvas, 
                     (margin - 20, start_y - 30),
                     (self.canvas_width - margin + 20, start_y + total_height),
                     (240, 240, 240), -1)
        
        cv2.rectangle(self.canvas, 
                     (margin - 20, start_y - 30),
                     (self.canvas_width - margin + 20, start_y + total_height),
                     (100, 100, 100), 2)
        
        # Draw each line
        y_pos = start_y
        for line in lines:
            # Calculate x position to center the text
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            x_pos = (self.canvas_width - text_width) // 2
            
            cv2.putText(self.canvas, line,
                       (x_pos, y_pos),
                       font, font_scale, self.current_color, font_thickness)
            y_pos += line_height
        
        print(f"✍️ Text written on canvas: {text}")
    
    def draw_voice_indicator(self, display_canvas):
        """Draw voice mode indicator"""
        if not self.voice_mode:
            return
        
        # Draw voice mode indicator
        indicator_x = self.canvas_width // 2 - 150
        indicator_y = 20
        
        if self.is_listening:
            # Pulsing red indicator when listening
            import time
            pulse = int(abs(np.sin(time.time() * 3) * 100))
            color = (pulse, pulse, 255)
            status_text = "🎤 LISTENING..."
        else:
            color = (0, 200, 0)
            status_text = "🎤 Voice Mode - Press V to speak"
        
        # Draw background
        cv2.rectangle(display_canvas,
                     (indicator_x - 10, indicator_y - 5),
                     (indicator_x + 310, indicator_y + 35),
                     (50, 50, 50), -1)
        
        cv2.rectangle(display_canvas,
                     (indicator_x - 10, indicator_y - 5),
                     (indicator_x + 310, indicator_y + 35),
                     color, 3)
        
        # Draw text
        cv2.putText(display_canvas, status_text,
                   (indicator_x, indicator_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Main loop"""
        print("="*60)
        print("TOUCHLESS WRITING SYSTEM")
        print("="*60)
        print("\nGestures:")
        print("  👆 Index finger only     - Move cursor")
        print("  ✍️  Index + Thumb         - Draw")
        print("  🧹 Index + Middle        - Erase")
        print("  👌 Index + Pinky         - Select color")
        print("  ✋ All fingers           - Clear page")
        print("\nKeyboard shortcuts:")
        print("  N - New page")
        print("  P - Previous page")
        print("  → - Next page")
        print("  S - Save current page")
        print("  A - Save all pages")
        print("  T - Convert page to text (OCR)")
        print("  L - Live text mode (show text on page)")
        print("  V - Voice to text (speak and write)")
        print("  ↑/↓ - Scroll text (in live mode)")
        print("  C - Clear page")
        print("  Q - Quit")
        print("="*60)
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(frame_rgb)
            
            # Create display canvas
            display_canvas = self.canvas.copy()
            
            # Draw UI elements
            self.draw_status_bar(display_canvas)
            self.draw_color_palette(display_canvas)
            self.draw_instructions(display_canvas)
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get finger position
                    raw_position = self.get_finger_position(hand_landmarks, frame.shape)
                    
                    # Apply smoothing to position
                    position = self.smooth_position(raw_position)
                    
                    # Detect and stabilize gesture
                    detected_gesture = self.detect_gesture(hand_landmarks)
                    self.current_gesture = self.stabilize_gesture(detected_gesture)
                    
                    # Handle gesture
                    self.handle_gesture(self.current_gesture, position)
                    
                    # Draw cursor
                    self.draw_cursor(display_canvas, position, self.current_gesture)
                    
                    # Update status based on gesture
                    if self.message_timer > 0:
                        self.message_timer -= 1
                    else:
                        gesture_messages = {
                            'move': '👆 Moving cursor',
                            'draw': '✍️ Drawing...',
                            'erase': '🧹 Erasing...',
                            'select': '👌 Selection mode',
                            'clear': '✋ Clear page',
                            'idle': 'Show hand gestures'
                        }
                        self.status_message = gesture_messages.get(self.current_gesture, 'Ready')
                    
                    # Draw hand landmarks on webcam frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            else:
                self.current_gesture = 'idle'
                self.status_message = 'No hand detected'
                # Drop stale tracking history so the cursor resumes cleanly.
                self.reset_smoothing()
                self.last_position = None
            
            # Resize webcam frame for corner display
            small_frame = cv2.resize(frame, (320, 240))
            
            # Place webcam in corner
            display_canvas[self.canvas_height - 260:self.canvas_height - 20, 
                          self.canvas_width - 340:self.canvas_width - 20] = small_frame
            
            # Draw border around webcam
            cv2.rectangle(display_canvas, 
                         (self.canvas_width - 340, self.canvas_height - 260),
                         (self.canvas_width - 20, self.canvas_height - 20),
                         (255, 255, 255), 2)
            
            # Draw live text overlay if in live text mode
            self.draw_live_text_overlay(display_canvas)
            
            # Draw voice mode indicator
            self.draw_voice_indicator(display_canvas)
            
            # Show the display
            cv2.imshow('Touchless Writing System', display_canvas)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.new_page()
            elif key == ord('p'):
                self.previous_page()
            elif key == 83:  # Right arrow
                self.next_page()
            elif key == 82:  # Up arrow
                self.scroll_text('up')
            elif key == 84:  # Down arrow
                self.scroll_text('down')
            elif key == ord('s'):
                self.save_image()
            elif key == ord('a'):
                self.save_all_pages()
            elif key == ord('t'):
                self.convert_page_to_text()
            elif key == ord('l'):
                self.toggle_live_text_mode()
            elif key == ord('v'):
                if not self.voice_mode:
                    self.toggle_voice_mode()
                else:
                    self.start_voice_recording()
            elif key == 27:  # ESC key
                if self.voice_mode:
                    self.toggle_voice_mode()
            elif key == ord('c'):
                self.clear_page()
            elif key == ord('x'):
                self.use_preprocessing = not self.use_preprocessing
                state = "ON" if self.use_preprocessing else "OFF"
                self.set_status(f"OCR preprocessing {state}")
                print(f"OCR preprocessing {state}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\nApplication closed. Thank you!")


if __name__ == "__main__":
    app = TouchlessWritingSystem()
    app.run()