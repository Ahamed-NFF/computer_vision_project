# 🎤 VOICE TO TEXT - Speech Recognition Feature

## Overview
The Voice to Text feature allows you to dictate text using your voice and have it automatically written on the canvas. Perfect for quick note-taking, accessibility, and hands-free operation!

## 🚀 How to Use

### Basic Workflow
1. **Press `V`** - Activate Voice Mode
2. **Press `V` again** - Start listening/recording
3. **Speak clearly** into your microphone
4. **Wait** - Text will be automatically written on canvas
5. **Press `ESC`** - Exit Voice Mode

### Detailed Steps

#### Step 1: Enable Voice Mode
```
Press 'V' key → Voice Mode indicator appears at top center
```

#### Step 2: Start Recording
```
Press 'V' again → Red pulsing indicator shows "LISTENING..."
```

#### Step 3: Speak
```
Speak your text clearly → Wait for processing (2-5 seconds)
```

#### Step 4: See Results
```
Text appears on canvas → Formatted and centered automatically
```

## ✨ Features

### 🎯 Smart Text Placement
- **Centered layout** - Text appears in the middle of the canvas
- **Auto word wrap** - Long text automatically wraps to multiple lines
- **Background box** - Light gray background for better readability
- **Clean formatting** - Professional appearance with proper spacing

### 🎨 Visual Indicators
- **Green indicator** - Voice mode active, ready to listen
- **Red pulsing** - Currently listening to your speech
- **Status updates** - Real-time feedback in status bar

### 🔊 Speech Recognition
- **Google Speech API** - Industry-leading accuracy
- **Multiple languages** - Supports various languages (English default)
- **Ambient noise filtering** - Automatically adjusts for background noise
- **15-second recording** - Captures up to 15 seconds of speech
- **10-second timeout** - Stops if no speech detected

## 🎮 Controls

| Key | Action |
|-----|--------|
| **V** (first press) | Enable Voice Mode |
| **V** (second press) | Start voice recording |
| **ESC** | Exit Voice Mode |

## 📋 Voice Mode States

### State 1: Inactive
- No voice indicator visible
- Press `V` to activate

### State 2: Active (Ready)
- Green indicator: "🎤 Voice Mode - Press V to speak"
- Press `V` to start listening

### State 3: Listening
- Red pulsing indicator: "🎤 LISTENING..."
- Speak now!
- System is recording

### State 4: Processing
- Status shows: "Processing speech..."
- AI is converting speech to text
- Wait a moment

### State 5: Complete
- Status shows: "✅ Recognized: [your text]"
- Text appears on canvas
- Ready for next recording

## 🎯 Use Cases

### 📝 Quick Notes
- Dictate ideas quickly without typing
- Capture thoughts while drawing
- Add annotations to sketches

### ♿ Accessibility
- Hands-free text input
- Alternative input method
- Useful for motor disabilities

### 🏃 Fast Documentation
- Speed up note-taking
- Combine drawing + voice notes
- Create mixed-media presentations

### 🌍 Multilingual Support
- Speak in different languages
- Automatic language detection
- Great for language learning

## ⚙️ Technical Details

### Speech Recognition Engine
- **Provider:** Google Speech Recognition API
- **Method:** Cloud-based processing
- **Accuracy:** High (industry-leading)
- **Internet:** Required

### Audio Settings
- **Microphone:** Default system microphone
- **Sample Rate:** Auto-detected
- **Noise Reduction:** Automatic (0.5s ambient noise adjustment)
- **Timeout:** 10 seconds (no speech)
- **Max Duration:** 15 seconds per recording

### Text Rendering
- **Font:** OpenCV Hershey Simplex
- **Size:** 0.8 scale
- **Color:** Uses current selected color
- **Position:** Auto-centered
- **Wrapping:** Automatic word wrap

## 💡 Tips for Best Results

### 🎤 Microphone Tips
1. **Use a good microphone** - Built-in or external USB mic
2. **Position correctly** - 6-12 inches from mouth
3. **Reduce background noise** - Quiet environment works best
4. **Check permissions** - Allow microphone access in Windows

### 🗣️ Speaking Tips
1. **Speak clearly** - Natural pace, not too fast
2. **Good volume** - Normal speaking volume
3. **Minimize pauses** - Long pauses may trigger timeout
4. **Punctuation** - Say "comma", "period", "question mark" for punctuation

### 📝 Content Tips
1. **Short phrases work best** - 1-3 sentences at a time
2. **Pause between recordings** - Wait for text to appear
3. **Check recognition** - Verify text is correct
4. **Repeat if needed** - Re-record if not recognized

## 🚨 Troubleshooting

### "No microphone found"
**Solution:**
- Check microphone is connected
- Allow microphone permission in Windows settings
- Try different USB port (external mic)
- Restart application

### "Timeout - No speech detected"
**Solution:**
- Speak louder
- Move microphone closer
- Reduce background noise
- Check microphone is working

### "Could not understand audio"
**Solution:**
- Speak more clearly
- Reduce speech speed
- Minimize background noise
- Check microphone quality
- Try again with simpler words

### "Speech recognition error"
**Solution:**
- Check internet connection (required!)
- Verify API access
- Try again after a moment
- Restart application if persistent

### Text not appearing on canvas
**Solution:**
- Check if canvas has space
- Verify voice mode is active
- Look for status messages
- Try clearing page first

## 📦 Installation Requirements

### Python Packages
```bash
pip install SpeechRecognition
pip install PyAudio
```

### Windows Additional Setup
If PyAudio fails to install:
```bash
# Download wheel file from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

### Microphone Permissions
1. Open Windows Settings
2. Privacy & Security → Microphone
3. Enable "Let apps access your microphone"
4. Enable for Python

## 🔄 Workflow Examples

### Example 1: Quick Note
```
1. Draw a diagram
2. Press V (enable voice mode)
3. Press V (start listening)
4. Say: "This is the main component"
5. Text appears on canvas
6. Continue drawing
```

### Example 2: Multiple Notes
```
1. Press V (enable)
2. Press V (speak) → "First point"
3. Wait for text
4. Press V (speak) → "Second point"
5. Wait for text
6. Press ESC (exit voice mode)
```

### Example 3: Mixed Media
```
1. Draw with gestures
2. Press V twice → Add voice note
3. Press L → Check with live text mode
4. Press T → Save as text file
5. Press S → Save as image
```

## 🎨 Integration with Other Features

### Works With:
- ✅ All drawing gestures
- ✅ Color selection
- ✅ Multi-page support
- ✅ Image saving
- ✅ Text extraction (OCR)
- ✅ Live text mode

### Combine Features:
1. **Voice + Drawing** - Annotate sketches with voice
2. **Voice + OCR** - Convert voice text to digital text
3. **Voice + Multi-page** - Add voice notes to multiple pages
4. **Voice + Color** - Use different colors for voice text

## 🌟 Advanced Features

### Continuous Recording Mode
Press `V` multiple times to keep adding voice notes without exiting voice mode!

### Text Accumulation
Voice text is added to canvas, doesn't replace existing content.

### Background Processing
Speech recognition runs in separate thread, doesn't freeze UI.

### Error Recovery
Automatic error handling with helpful status messages.

## 📊 Comparison

| Feature | Voice to Text | Gesture Drawing | Live OCR |
|---------|--------------|-----------------|----------|
| Input | Voice | Hand gestures | Image |
| Speed | Fast | Medium | Slow |
| Accuracy | High | Manual | High |
| Use Case | Text notes | Drawings | Convert writing |
| Internet | Required | Not needed | Required |

## 🎯 Best Practices

1. **Enable voice mode when needed** - Don't leave it on always
2. **Speak one thought at a time** - Better accuracy
3. **Check text after each recording** - Verify correctness
4. **Use with other features** - Combine for best results
5. **Exit when done** - Press ESC to disable voice mode

---

## 🎤 Quick Reference Card

```
┌─────────────────────────────────────┐
│      VOICE TO TEXT CONTROLS         │
├─────────────────────────────────────┤
│  V     → Toggle Voice Mode          │
│  V     → Start Recording (in mode)  │
│  ESC   → Exit Voice Mode            │
├─────────────────────────────────────┤
│  🟢 Green = Ready                   │
│  🔴 Red Pulse = Listening           │
│  ✅ Check = Success                 │
└─────────────────────────────────────┘
```

**Start using voice input today and experience hands-free note-taking! 🎤✨**
