# 🔴 LIVE TEXT MODE - Real-time OCR Display

## Overview
The Live Text Mode feature allows you to instantly convert your handwritten notes to text and display it as an overlay on the same page - no need to open external files!

## 🚀 How to Use

### Activate Live Text Mode
1. Draw or write on your canvas using hand gestures
2. Press **`L`** on your keyboard
3. Wait a moment while the AI processes your handwriting
4. The extracted text appears as an overlay on top of your drawing!

### Navigate the Text
- **Press `L` again** - Toggle live text mode ON/OFF
- **Up Arrow ↑** - Scroll text up
- **Down Arrow ↓** - Scroll text down

## ✨ Features

### 🎨 Beautiful Overlay Design
- **Semi-transparent dark background** - Easy to read without completely hiding your drawing
- **Cyan border** - Clear visual indicator of the overlay
- **Scrollbar** - Shows your position in longer texts
- **Line counter** - Displays current lines visible (e.g., "1-20/45")

### 📜 Smart Text Display
- **Automatic word wrapping** - Text fits perfectly within the overlay
- **Smooth scrolling** - Navigate through long texts easily
- **Large, readable font** - Optimized for clarity
- **Status updates** - Shows when processing or in live mode

### 🎯 Use Cases
- **Quick review** - Instantly check what you wrote without saving files
- **Real-time verification** - See if your handwriting is being recognized correctly
- **Presentation mode** - Show both handwriting and typed text simultaneously
- **Note comparison** - Compare your handwriting with the extracted text side-by-side

## 🎮 Controls Summary

| Key | Action |
|-----|--------|
| **L** | Toggle Live Text Mode ON/OFF |
| **↑** | Scroll text up (in live mode) |
| **↓** | Scroll text down (in live mode) |
| **T** | Save text to file (traditional OCR) |

## 🆚 Live Mode vs. File Mode

### Live Text Mode (Press L)
- ✅ Instant visual feedback on screen
- ✅ No files created
- ✅ Compare text with drawing
- ✅ Perfect for quick checks
- ⚠️ Temporary - disappears when you press L again

### File Mode (Press T)
- ✅ Saves text to `.txt` file
- ✅ Permanent storage
- ✅ Can be opened in any text editor
- ✅ Perfect for archiving
- ✅ Includes timestamp and metadata

**Pro Tip:** Use both! Press `L` for quick review, then press `T` to save if you want to keep it.

## 🎨 Visual Indicators

### When Active:
- **Dark overlay** with cyan border appears
- **Title bar** says "EXTRACTED TEXT (Live Mode)"
- **Status bar** shows "Live Text Mode - Press L to exit | ↑↓ to scroll"
- **Scrollbar** appears on the right if text is longer than screen

### When Inactive:
- Normal canvas view
- No overlay visible
- Full access to drawing tools

## 💡 Tips for Best Results

1. **Write clearly** - The AI recognizes clear handwriting better
2. **Use dark colors** - Black or dark blue on white background works best
3. **Good spacing** - Don't cram words together
4. **Wait for processing** - Give it 2-3 seconds to analyze
5. **Check in live mode first** - Before saving, verify text is correct

## 🔧 Technical Details

### Processing Flow:
1. Current canvas is saved to temporary file
2. Image sent to OpenAI GPT for OCR
3. Text extracted and displayed in overlay
4. Temporary file deleted automatically

### Performance:
- **Processing time:** 2-5 seconds (depends on internet speed)
- **Text capacity:** Unlimited (with scrolling)
- **Memory efficient:** Only stores text, not duplicate images

### Requirements:
- Active internet connection (for OpenAI GPT)
- OpenAI API key in `.env` file
- Same requirements as the file-based OCR

## 🎯 Example Workflow

### Quick Note Review:
```
1. Write notes with gestures → 
2. Press L (live mode) → 
3. Read extracted text → 
4. Press L again (exit) → 
5. Continue writing
```

### Save Important Notes:
```
1. Write notes with gestures → 
2. Press L (preview in live mode) → 
3. Scroll to verify all text → 
4. Press T (save to file) → 
5. Press L (exit live mode)
```

## 🚨 Troubleshooting

### "Converting to text... Please wait" stays too long
- Check your internet connection
- Verify API key in `.env` file
- Try pressing L again to cancel

### Text is cut off or not visible
- Use Up/Down arrows to scroll
- Check the line counter at bottom right
- Text might be longer than visible area

### Text extraction is incorrect
- Write more clearly
- Use darker colors
- Ensure good lighting for camera
- Try writing larger

## 🎊 Benefits

✅ **Instant feedback** - See results immediately  
✅ **Non-destructive** - Your drawing stays intact  
✅ **Toggle on/off** - Switch between views easily  
✅ **Perfect for learning** - See how AI interprets your writing  
✅ **Presentation ready** - Show both formats at once  

---

**Press `L` to enter the future of handwriting conversion! ✨**
