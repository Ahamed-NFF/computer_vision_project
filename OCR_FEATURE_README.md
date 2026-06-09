# OCR Feature - Image to Text Conversion

## Overview
The Touchless Writing System now includes OCR (Optical Character Recognition) functionality that can extract text from your handwritten notes using OpenAI's GPT.

## How to Use

### Keyboard Shortcut
Press **`T`** while the application is running to convert the current page to text.

### What Happens
1. The current page/canvas is saved as a temporary image
2. The image is sent to OpenAI GPT for text extraction
3. Extracted text is saved to a file in the `extracted_text/` folder
4. The text is also printed to the console
5. A status message confirms the operation

## Output Files

### Location
All extracted text files are saved in: `extracted_text/`

### File Format
```
extracted_text/page_1_20251007_143022.txt
```

### Content Structure
```
Extracted Text from Page 1
Date: 2025-10-07 14:30:22
============================================================

[Your extracted text appears here]
```

## Requirements

### Dependencies
- `openai` (OpenAI SDK)
- `python-dotenv` (for API key management)
- `Pillow` (PIL for image handling)

### API Key Setup
You need an OpenAI API key in your `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Get your API key from: https://platform.openai.com/api-keys

## Features

✅ **Accurate OCR** - Uses OpenAI's GPT-4o mini vision model  
✅ **Handwriting Recognition** - Can recognize handwritten text  
✅ **Multiple Languages** - Supports various languages  
✅ **Auto-Save** - Automatically saves extracted text to file  
✅ **Console Output** - Displays extracted text immediately  
✅ **Timestamped Files** - Each extraction is saved with a unique timestamp  

## Workflow Example

1. Draw or write on the canvas using hand gestures
2. When done, press **`T`** on your keyboard
3. Wait a moment for the AI to process
4. Check the console for extracted text
5. Find the saved text file in `extracted_text/` folder

## Folders Created

The feature automatically creates these folders if they don't exist:
- `temp/` - Temporary storage for processing
- `extracted_text/` - Permanent storage for all extracted text files

## Error Handling

If OCR fails:
- Error message shown in status bar
- Error details printed to console
- Temporary files are cleaned up automatically

## Tips for Best Results

1. **Write clearly** - Clear, legible writing works best
2. **Good contrast** - Use dark colors on white background
3. **Adequate size** - Write large enough for the AI to recognize
4. **Wait for processing** - Give the AI a few seconds to process

## Integration

The feature seamlessly integrates with:
- ✅ Multi-page support (extracts text from current page)
- ✅ Page navigation (works on any page)
- ✅ Color selection (works with any drawing color)
- ✅ All drawing gestures

## Keyboard Shortcuts Summary

| Key | Function |
|-----|----------|
| N | New page |
| P | Previous page |
| → | Next page |
| S | Save current page as image |
| A | Save all pages as images |
| **T** | **Convert page to text (OCR)** ⭐ |
| C | Clear page |
| Q | Quit |

---

**Note:** Make sure your `.env` file contains a valid OpenAI API key for the OCR feature to work.
