# Touchless Writing System — Presentation Script

> Speaker script (talk track) for **Touchless_Writing_System.pptx** — 14 slides.
> Image Processing & Computer Vision course project.
> Presented by **Mohammed Farees Mohammed Rifath**.

**How to use this:** each section below maps to one slide. The *"On screen"* line
reminds you what the audience sees; the *"Say"* block is what you speak. Total
target: **10–12 minutes** plus Q&A. Keep one finger on the spacebar — advance on
the cue words shown in **bold**.

---

## Slide 1 — Cover: "Touchless Writing System"

**On screen:** Title, subtitle, your name, and tech tags (OpenCV · MediaPipe · NumPy · OpenAI GPT-4o · Python 3.11).

**Say:**
> "Good morning / afternoon, everyone. My project is the **Touchless Writing
> System** — an air-writing application for the Image Processing and Computer
> Vision course.
>
> The idea in one sentence: **you write in the air with your hand, a webcam
> tracks it, and the system turns your handwriting into editable digital text** —
> no stylus, no touch screen, no special hardware. It also supports voice
> dictation and live on-screen text recognition.
>
> I built it in Python using OpenCV, MediaPipe for hand tracking, and a
> vision-language model for the handwriting recognition. Let me walk you through
> how it works and how well it performs."

⏱ ~45 sec. **Advance on "how well it performs."**

---

## Slide 2 — Agenda

**On screen:** Six numbered cards — Problem, Literature Review, Data Acquisition, Methodology, Results, Novelty & Conclusion.

**Say:**
> "Here's the roadmap. I'll start with **the problem** — why touch-free input
> matters. Then a quick **literature review** of how others have approached
> air-writing and OCR. I'll cover **how I capture the input data**, then the
> **methodology** — the full pipeline from webcam frame to text. After that the
> **evaluation results** — character and word error rates and latency — and I'll
> close with the **novelty, conclusions, and future work**."

⏱ ~30 sec. Keep this brisk — don't read all six aloud, just signpost.

---

## Slide 3 — Project Problem

**On screen:** Four problem cards (hygiene/accessibility, hardware cost, natural interaction, hand-off to digital) + the goal strip at the bottom.

**Say:**
> "Conventional digital writing always needs a **physical surface** — a keyboard,
> a stylus, or a trackpad. That's limiting in a lot of real settings.
>
> First, **hygiene and accessibility**: shared touch devices spread germs, and
> some users simply can't operate a stylus comfortably. Second, **cost**: digital
> pens, smart-boards, and touch panels are expensive and often unavailable. Third,
> there's **no natural, hardware-light way** to write or annotate in the air — say,
> while you're teaching or presenting. And fourth, even when you do capture notes,
> **turning handwriting into editable, searchable text is a separate manual step**.
>
> So the goal" — *(point to the strip)* — "is a **webcam-only system** that lets
> you write in the air with hand gestures and convert that handwriting straight to
> digital text."

⏱ ~55 sec. **Advance on the goal sentence.**

---

## Slide 4 — Aim & Objectives

**On screen:** "Project Aim" panel on the left; six numbered objectives on the right.

**Say:**
> "The **aim** is on the left: build a touchless writing app that lets you draw in
> the air using only a standard webcam, and convert that handwriting into editable
> text using computer-vision pre-processing and OCR.
>
> That breaks into six **objectives**: one, **real-time hand tracking** — detect
> 21 hand landmarks and recognise finger gestures. Two, a **gesture-driven canvas**
> — map gestures to move, draw, erase, select and clear. Three, a **CV
> pre-processing pipeline** to clean the strokes before recognition. Four, the
> **handwriting-to-text OCR** itself. Five, **multimodal extras** — voice dictation
> and a live on-screen text overlay. And six — importantly — **objective
> evaluation**: measuring OCR accuracy and latency, with and without
> pre-processing."

⏱ ~55 sec. Stress objective six — it's what makes this rigorous, not just a demo.

---

## Slide 5 — Literature Review

**On screen:** Four columns — sensor/glove, color-marker, deep-learning landmarks, OCR/HTR — each with a pro (✓) and con (✗); positioning line at the bottom.

**Say:**
> "There are four broad families of prior work.
>
> **Sensor- or glove-based** approaches — Leap Motion, data-gloves, Kinect depth —
> are very accurate, but they need extra hardware and aren't webcam-friendly.
>
> **Color-marker tracking** is the classic OpenCV air-canvas: you track a colored
> fingertip with HSV thresholding. It's cheap and real-time, but you need a marker
> and it's fragile to lighting.
>
> **Deep-learning landmark** estimators — MediaPipe Hands, OpenPose — give you 21
> keypoints from plain RGB. Marker-free, robust, fast; the trade-off is you depend
> on a trained model.
>
> And for the text side, **OCR and handwriting-recognition engines**: Tesseract is
> mature but weak on free handwriting, whereas vision-LLMs read it far better.
>
> So **my positioning**" — *(point to bottom)* — "is to combine marker-free
> MediaPipe landmark tracking for the input, a classical CV pre-processing
> pipeline, and a vision-LLM for OCR — all **webcam-only, no special hardware**."

⏱ ~70 sec. **Advance on "no special hardware."**

---

## Slide 6 — Data Acquisition

**On screen:** Left = 5-step capture flow; right = "Acquisition Specs" panel.

**Say:**
> "So how does data actually enter the system?
>
> It starts with the **webcam stream** — live RGB frames at 1280 by 720, captured
> with OpenCV. Each frame goes to **MediaPipe Hands**, which extracts **21 (x, y, z)
> keypoints** for a single hand. From those landmarks I compute each finger's
> up/down state and turn it into a discrete **gesture signal**, stabilised over
> three frames so it doesn't flicker. The cursor trail is rendered onto a **NumPy
> canvas** — that's the 'page' image. And in parallel there are two extra inputs:
> **microphone audio** for voice mode, and a small **labelled image-and-text set**
> for the OCR evaluation.
>
> On the right are the specs — the key point is it's a **standard laptop or USB
> webcam, no depth sensor, no glove**."

⏱ ~60 sec. You can paraphrase the specs panel rather than read each row.

---

## Slide 7 — System Architecture (End-to-End Pipeline)

**On screen:** Six-stage horizontal pipeline (Frame → Tracking → Gesture → Canvas → Pre-process → OCR) + two branch boxes (Voice path, Live text overlay).

**Say:**
> "Here's the whole pipeline end-to-end. A **webcam frame** comes in; **MediaPipe**
> turns it into 21 landmarks; **gesture detection** converts finger states into an
> action; that action draws onto the **digital canvas**; when you ask for text, the
> page goes through **CV pre-processing** — contrast, thresholding, morphology, crop
> — and finally into the **OCR engine**, GPT-4o-mini, which returns text.
>
> Two branches sit alongside the main path. The **voice path**: microphone audio
> goes through Google Speech Recognition and the text is rendered straight onto the
> canvas — and it runs on a background thread so the UI never freezes. And the
> **live text overlay**: the same OCR, but the recognised text is drawn back onto
> the page as a semi-transparent overlay for instant verification, with no file
> saved."

⏱ ~60 sec. This is the spine of the talk — point along the arrows as you go.

---

## Slide 8 — Hand-Gesture Recognition

**On screen:** Six gesture cards (Move, Draw, Erase, Select, Clear, Idle) + the "Why classical" line.

**Say:**
> "Let me zoom into the gesture recognition, because it's the core CV interaction.
>
> For each finger I compare the **fingertip to its lower joint** to decide whether
> it's up or down. The pattern of raised fingers maps to one gesture. **Index only**
> is **Move** — the cursor follows your finger with no ink. **Index plus thumb** is
> **Draw**. **Index plus middle** is **Erase**. **Index plus pinky** is **Select**,
> for picking a colour. **All five fingers** **Clears** the page, and anything else
> is **Idle**.
>
> And the design choice at the bottom matters" — *(point)* — "I deliberately used
> **geometric finger-state logic, not a trained classifier**, because it's
> interpretable, training-free, and runs in real time on a CPU."

⏱ ~60 sec. If short on time, name only Move/Draw/Erase and gesture at the rest.

---

## Slide 9 — OCR Pre-processing Pipeline (Core CV)

**On screen:** Six step-cards + the pipeline strip at the bottom.

**Say:**
> "This slide is the **classical computer-vision heart** of the project — the
> pre-processing applied before OCR. Every stage is **togglable**, which is what
> lets me run the A/B evaluation later.
>
> Step one, **grayscale** — drop the colour channels. Two, **CLAHE** — adaptive
> histogram equalisation, which is robust to uneven lighting on the canvas. Three,
> **adaptive thresholding** — Gaussian binarisation with block size 31 — to get
> clean binary strokes. Four, **morphology** — a close to reconnect broken strokes
> and an open to remove speckle noise. Five, **crop to the ink** — a bounding box
> around the written content so OCR isn't distracted by empty space. And six,
> **invert back** to natural black-ink-on-white.
>
> So the full chain is" — *(read the strip)* — "**grayscale, CLAHE, adaptive
> threshold, morphology, crop, invert.**"

⏱ ~70 sec. **Advance after reading the strip.**

---

## Slide 10 — Recognition & Implementation

**On screen:** Two cards (OCR, Voice-to-Text) on the left; "Implementation Stack" panel on the right.

**Say:**
> "For **recognition**: the cleaned page is base64-encoded and sent to **OpenAI
> GPT-4o-mini**, a vision model, which returns editable text — saved to a
> timestamped file, or shown live. I chose a vision-LLM specifically because
> **Tesseract struggles with free-form air-handwriting**.
>
> The **voice-to-text** path transcribes microphone audio with Google Speech
> Recognition on a background thread, then word-wraps and renders it onto the canvas
> — a fast alternative when you have a long passage to enter.
>
> The full stack is on the right: **Python 3.11, OpenCV, MediaPipe 0.10.21, NumPy,
> the OpenAI SDK, SpeechRecognition, dotenv** for the API key, and a **custom
> evaluation harness** I wrote with no external metric libraries."

⏱ ~55 sec.

---

## Slide 11 — Results & Discussion

**On screen:** Four metric cards (CER, WER, Latency, Pre-proc gain) + "What worked" / "Limitations" bullets + the *Note* footnote.

> ⚠️ **Honesty note for the presenter — read before you speak this slide.**
> The metric cards on the slide show the **acceptance *targets*** (CER ≤ 0.15,
> WER ≤ 0.25, ≈ 3–5 s). The **actual numbers** from `eval_results/summary.txt` on
> the current small demo set are:
> - **Mean CER ≈ 0.18** (with and without pre-processing — essentially tied)
> - **Mean WER ≈ 1.0** (very high — the tiny 2-sample set has short references,
>   so a few wrong words dominates the rate)
> - **Mean latency ≈ 1.0 s with pre-processing vs ≈ 1.65 s without**
>
> Don't claim the targets were hit. Present them as targets, then state your real
> findings. The defensible headline is: **pre-processing did *not* worsen CER and
> it *reduced latency*; the WER is inflated by a very small dataset.** If you can,
> re-run `python evaluate.py` on 5–10 pages before the talk to get a fairer WER.

**Say (honest version):**
> "Now the evaluation. I measured OCR accuracy with **Character and Word Error
> Rate**, plus latency, in an **A/B design** — with versus without the CV pipeline.
> The cards here show my **acceptance targets**.
>
> On my current demo set, the **character error rate came in around 0.18**, and
> crucially it was the **same with and without pre-processing** — so the pipeline
> **didn't hurt accuracy**, and it actually **cut latency** from about 1.6 seconds
> down to around 1 second per page. The **word error rate is high** on this set,
> but that's a small-dataset artefact — with very short reference texts, a single
> wrong word spikes the rate; I'd expect that to settle with more samples.
>
> What worked: **real-time tracking on CPU with no GPU**, and the **vision-LLM
> reading free handwriting far better than Tesseract**. The **limitations** are
> honest ones — **poor lighting degrades the landmark tracking**, **OCR and voice
> need internet and an API key**, and I track a **single hand**, so very messy
> writing lowers accuracy."

⏱ ~80 sec. This is the slide examiners probe — owning the real numbers builds credibility.

---

## Slide 12 — Novelty & Key Contributions

**On screen:** Four dark cards (hardware-free multimodal, classical+modern, built-in evaluation, real-time on CPU) + the one-line summary.

**Say:**
> "So what's **novel** here? Four things.
>
> One — it's **hardware-free and multimodal**: air-writing, voice, and live OCR in
> a **single webcam-only app**, with no glove, marker, or touch surface. Two — it
> **bridges classical and modern CV**: an interpretable, togglable pre-processing
> pipeline feeding a vision-LLM — the best of both. Three — and this is rare in a
> student demo — it has a **built-in objective evaluation harness** with a proper
> A/B study. And four — it runs **in real time on a commodity CPU**, with the OCR
> and voice off the UI thread so it never freezes.
>
> In one line" — *(point)* — "**a webcam becomes a touchless smart-notebook you can
> write to, speak to, and read back from.**"

⏱ ~55 sec.

---

## Slide 13 — Conclusion & Future Work

**On screen:** "Conclusion" panel (left) + "Future Work" list (right).

**Say:**
> "To **conclude**: I delivered a working touchless writing system using **only a
> webcam**. Its CV core is **MediaPipe tracking plus a classical pre-processing
> pipeline**; it **digitises both handwriting and speech** into editable text; and
> I **validated it objectively** with CER, WER and latency. The impact is a
> **hygienic, low-cost, accessible, and intuitive** input method.
>
> For **future work**: moving to **on-device offline OCR** to drop the network
> dependency; **two-hand and multi-user** support for collaborative whiteboarding;
> **shape and equation recognition**; **gesture customisation with undo/redo**; and
> **export to searchable PDF with cloud sync**."

⏱ ~55 sec.

---

## Slide 14 — Thank You / Q&A

**On screen:** "Thank You", subtitle, your name, and a "Q & A" badge.

**Say:**
> "That's the Touchless Writing System — **write in the air, get digital text.**
> Thank you for listening; I'm happy to take any questions."

⏱ ~10 sec.

---

## Appendix — Anticipated Q&A (keep in your back pocket)

**Q: Why a cloud vision-LLM instead of Tesseract or an on-device model?**
> Tesseract is built for printed text and does poorly on free-form air-handwriting,
> which is exactly my input. A vision-LLM reads it far more reliably. I list
> on-device HTR as future work to remove the network dependency.

**Q: Your WER is 1.0 — isn't that a failure?**
> It's a small-dataset artefact. With very short reference strings, one wrong word
> dominates the rate. The more stable signals are that pre-processing didn't worsen
> CER and reduced latency. A larger eval set would give a fairer WER.

**Q: Why does pre-processing not improve CER here?**
> The vision-LLM already handles clean canvas images well, so on clean synthetic
> pages there's little left to gain. Pre-processing still helps for noisy, real-world
> capture, and it measurably cut latency by cropping to the ink region.

**Q: Why finger-state geometry instead of a gesture classifier?**
> It's interpretable, needs no training data, and runs in real time on CPU. For six
> well-separated gestures, a heavy classifier would be over-engineering.

**Q: How do you stop gesture flicker?**
> The detected gesture is stabilised over three consecutive frames before it's
> accepted, which removes single-frame noise.

**Q: What are the privacy implications of sending images to OpenAI?**
> Only the rendered canvas page is sent — never the raw webcam feed — and only when
> the user explicitly presses the OCR key. On-device OCR (future work) would remove
> the cloud round-trip entirely.

---

### Timing summary

| Section | Slides | Approx. |
|---|---|---|
| Intro & framing | 1–4 | ~3 min |
| Background & data | 5–6 | ~2 min |
| Methodology | 7–10 | ~4 min |
| Results & close | 11–14 | ~3 min |
| **Total** | **14** | **~12 min** + Q&A |
</content>
</invoke>
