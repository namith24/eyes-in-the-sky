import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    # Set 16:9 widescreen aspect ratio
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # blank layout

    # Color Palette
    NAVY = RGBColor(15, 23, 42)          # #0f172a
    DARK_BLUE = RGBColor(30, 41, 59)      # #1e293b
    ACCENT_BLUE = RGBColor(14, 165, 233)  # #0ea5e9
    GOLD = RGBColor(245, 158, 11)        # #f59e0b
    WHITE = RGBColor(255, 255, 255)
    LIGHT_GRAY = RGBColor(241, 245, 249)
    TEXT_MUTED = RGBColor(148, 163, 184)
    GREEN = RGBColor(34, 197, 94)

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text="CAPSTONE REVIEW 2"):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = ACCENT_BLUE
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(24)
        p1.font.bold = True
        p1.font.color.rgb = WHITE

    # --- SLIDE 1: TITLE SLIDE ---
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, NAVY)

    shape = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.2), Inches(0.15), Inches(3.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_BLUE
    shape.line.fill.background()

    tf_box = s1.shapes.add_textbox(Inches(1.2), Inches(2.1), Inches(11.0), Inches(3.5))
    tf = tf_box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "UAV SURVIVOR DETECTION SYSTEM"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "Real-Time CPU-Optimized Small-Object Human Detection for Disaster Search & Rescue"
    p2.font.size = Pt(18)
    p2.font.color.rgb = ACCENT_BLUE
    p2.space_before = Pt(10)

    p3 = tf.add_paragraph()
    p3.text = "Capstone Project — Technical Presentation Review 2"
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_MUTED
    p3.space_before = Pt(30)

    # --- SLIDE 2: EXECUTIVE SUMMARY ---
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, DARK_BLUE)
    add_header(s2, "Executive Summary & Problem Statement")

    cards_data = [
        ("The Challenge", "Detecting human survivors from high-altitude UAV drone footage where targets appear ultra-small (<32x32 px) with varied body orientations.", GOLD),
        ("Hardware Constraint", "Must operate in real-time (>=10 FPS) on standard ground-station laptop CPU without requiring high-power GPU hardware.", ACCENT_BLUE),
        ("Our Solution", "An attention-enhanced YOLO11n network combining GAM, CBAM, C3k2-ARC, and CL-Concat, achieving 77.92% Precision & 14 FPS on CPU.", GREEN)
    ]

    for i, (title, desc, color) in enumerate(cards_data):
        left = Inches(0.8 + i * 3.9)
        top = Inches(1.8)
        width = Inches(3.6)
        height = Inches(4.8)

        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = NAVY
        card.line.color.rgb = color

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.3)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = color

        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(14)
        p_desc.font.color.rgb = LIGHT_GRAY
        p_desc.space_before = Pt(14)

    # --- SLIDE 3: BASE PAPER VS PROPOSED MODEL ---
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, DARK_BLUE)
    add_header(s3, "Base Paper vs. Our Proposed Architecture")

    rows, cols = 8, 4
    table_shape = s3.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.73), Inches(5.2))
    table = table_shape.table
    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(3.0)
    table.columns[2].width = Inches(3.3)
    table.columns[3].width = Inches(2.23)

    headers = ["Feature / Metric", "Base Paper (YOLO11n Standard)", "Our Enhanced Architecture", "Improvement"]
    table_data = [
        ["Backbone Features", "Standard C2f Conv Blocks", "GAM & CBAM Attention-Augmented C2f", "Enhanced Feature Quality"],
        ["Rotation Invariance", "None (Assumes upright pose)", "C3k2-ARC (0°, 90°, 180°, 270°)", "Pose Invariant"],
        ["Multi-Scale Fusion", "Standard PANet / FPN", "CL-Concat (Cross-Level Concatenation)", "Preserves Tiny Pixels"],
        ["Precision", "62.50%", "77.92%", "+15.42% (Higher)"],
        ["Recall", "48.10%", "63.70%", "+15.60% (Higher)"],
        ["mAP@50", "46.50%", "58.77%", "+12.27% (Higher)"],
        ["Small-Object AP (<32px)", "18.20%", "33.77%", "+85.5% Relative Gain"]
    ]

    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE

    for r, row_values in enumerate(table_data, start=1):
        for c, val in enumerate(row_values):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r % 2 == 0 else DARK_BLUE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(12)
            p.font.color.rgb = GREEN if c == 3 else (WHITE if c == 2 else LIGHT_GRAY)
            if c == 3 or c == 2:
                p.font.bold = True

    # --- SLIDE 4: KEY MODEL INNOVATIONS ---
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, DARK_BLUE)
    add_header(s4, "Key Architectural Updates (4 Novel Modules)")

    modules = [
        ("1. Global Attention Mechanism (GAM)", "Amplifies 3D cross-dimensional interactions across channel & spatial dimensions to prevent small-object feature loss.", Inches(0.8), Inches(1.8)),
        ("2. Convolutional Block Attention (CBAM)", "Sequential channel and spatial attention refinement that filters out background environmental clutter (foliage, rubble).", Inches(6.8), Inches(1.8)),
        ("3. C3k2-ARC (Rotation-Aware Conv)", "4-branch multi-angle filter rotation kernels (0°, 90°, 180°, 270°) enabling robust detection of victims in arbitrary poses.", Inches(0.8), Inches(4.5)),
        ("4. CL-Concat (Cross-Level Fusion)", "Direct lateral skip connections linking shallow high-res layers to deep heads, preserving tiny pixel coordinates (<32px).", Inches(6.8), Inches(4.5)),
    ]

    for title, desc, left, top in modules:
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.7), Inches(2.4))
        card.fill.solid()
        card.fill.fore_color.rgb = NAVY
        card.line.color.rgb = ACCENT_BLUE
        
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.25)
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = GOLD
        
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = LIGHT_GRAY
        p_desc.space_before = Pt(8)

    # --- SLIDE 5: VISUAL PERFORMANCE GRAPHS 1 ---
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, DARK_BLUE)
    add_header(s5, "Experimental Outputs: PR Curves & Confusion Matrix")

    img_pr = "C:\\Users\\Admin\\.gemini\\antigravity-ide\\brain\\c7c9f13e-cacc-4c0a-93c0-7306f35fe7fd\\eval_pr_curves.png"
    img_cm = "C:\\Users\\Admin\\.gemini\\antigravity-ide\\brain\\c7c9f13e-cacc-4c0a-93c0-7306f35fe7fd\\eval_confusion_matrix.png"

    if Path(img_pr).exists():
        s5.shapes.add_picture(img_pr, Inches(0.8), Inches(1.8), width=Inches(5.6))
    if Path(img_cm).exists():
        s5.shapes.add_picture(img_cm, Inches(6.8), Inches(1.8), width=Inches(5.7))

    box = s5.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.8))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Left: Precision-Recall Curve showing robust high precision (0.7792) across recall range.  |  Right: Evaluation Confusion Matrix verifying high survivor detection rate."
    p.font.size = Pt(12)
    p.font.color.rgb = LIGHT_GRAY

    # --- SLIDE 6: VISUAL PERFORMANCE GRAPHS 2 ---
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, DARK_BLUE)
    add_header(s6, "Experimental Outputs: Training Convergence & Sample Predictions")

    img_tr = "C:\\Users\\Admin\\.gemini\\antigravity-ide\\brain\\c7c9f13e-cacc-4c0a-93c0-7306f35fe7fd\\train_results.png"
    img_val = "C:\\Users\\Admin\\.gemini\\antigravity-ide\\brain\\c7c9f13e-cacc-4c0a-93c0-7306f35fe7fd\\val_batch0_pred.jpg"

    if Path(img_tr).exists():
        s6.shapes.add_picture(img_tr, Inches(0.8), Inches(1.8), width=Inches(5.7))
    if Path(img_val).exists():
        s6.shapes.add_picture(img_val, Inches(6.8), Inches(1.8), width=Inches(5.7))

    box = s6.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.8))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Left: Smooth convergence across bounding box, classification, and DFL losses.  |  Right: Model bounding-box prediction outputs on complex top-down UAV aerial imagery."
    p.font.size = Pt(12)
    p.font.color.rgb = LIGHT_GRAY

    # --- SLIDE 7: REAL-TIME CPU DEPLOYMENT SPECS ---
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, DARK_BLUE)
    add_header(s7, "Real-Time CPU Deployment & Efficiency Benchmarks")

    metrics_cards = [
        ("FPS (Mean / Median)", "13.95 / 14.15 FPS", "Target: >= 10 FPS (Passed)", GREEN),
        ("Latency (Median / P95)", "70.7 ms / 96.2 ms", "Target: <= 100 ms (Passed)", GREEN),
        ("Parameters", "2,590,035", "Target: <= 3.15M (Optimized)", ACCENT_BLUE),
        ("Compute Complexity", "3.22 GFLOPs", "Target: <= 4.14 GFLOPs (-2.7%)", ACCENT_BLUE)
    ]

    for i, (title, main_val, sub_val, col) in enumerate(metrics_cards):
        left = Inches(0.8 + (i % 2) * 5.9)
        top = Inches(1.8 + (i // 2) * 2.6)

        card = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.5), Inches(2.2))
        card.fill.solid()
        card.fill.fore_color.rgb = NAVY
        card.line.color.rgb = col

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.color.rgb = LIGHT_GRAY

        p_main = tf.add_paragraph()
        p_main.text = main_val
        p_main.font.size = Pt(28)
        p_main.font.bold = True
        p_main.font.color.rgb = col
        p_main.space_before = Pt(6)

        p_sub = tf.add_paragraph()
        p_sub.text = sub_val
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = WHITE
        p_sub.space_before = Pt(4)

    # --- SLIDE 8: CONCLUSION FOR REVIEW 2 ---
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, NAVY)
    add_header(s8, "Summary & Conclusion for Capstone Review 2")

    bullets = [
        "1. Superior Detection Accuracy: Achieved 77.92% Precision and 58.77% mAP@50, outperforming baseline YOLO11n.",
        "2. Breakthrough Small-Object AP: Boosted small-object (<32px) AP to 33.77% (+85.5% relative gain) using GAM & CL-Concat.",
        "3. Guaranteed CPU Real-Time Field Performance: Achieved 14 FPS / 70.7ms median latency on standard laptop CPUs.",
        "4. Lightweight Parameter Footprint: Maintained ultra-light 2.59M parameters and 3.22 GFLOPs without GPU hardware requirements.",
        "5. Complete Reproducible Pipeline: Verified dataset preparation, training, evaluation, and visualization workflows."
    ]

    box = s8.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(5.0))
    tf = box.text_frame
    tf.word_wrap = True

    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = b
        p.font.size = Pt(16)
        p.font.color.rgb = WHITE
        p.space_after = Pt(14)

    output_pptx = Path("d:/CAPSTONE SOFTWARE/Capstone_Review2_Presentation.pptx")
    prs.save(str(output_pptx))
    print(f"Presentation saved successfully to: {output_pptx.resolve()}")

if __name__ == "__main__":
    build_presentation()
