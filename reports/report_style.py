"""Small python-docx layer for the plain report style (serif headings, sans body, light tables)."""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from docx_utils import _xslt, fix_table_widths, latex_to_mathml, set_cell_border  # noqa: E402

HEAD_FONT, BODY_FONT = "Georgia", "Calibri"
INK, MUTED, RULE, HEAD_FILL = RGBColor(0x1F, 0x1F, 0x1C), RGBColor(0x6A, 0x69, 0x65), "BDBBB5", "F3F2EE"
TEXT_W = 6.27   # A4 width minus 1" margins


def _field(run, code):
    for kind, text in (("begin", None), (None, code), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
        run._r.append(el)


class Report:
    def __init__(self, running_title):
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
        st = self.doc.styles["Normal"]
        st.font.name = BODY_FONT; st.font.size = Pt(11); st.font.color.rgb = INK
        st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.12
        r = s.header.paragraphs[0].add_run(running_title); r.font.size = Pt(8); r.font.color.rgb = MUTED
        fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for piece in ("Page ", "PAGE", " of ", "NUMPAGES"):
            r = fp.add_run(piece if " " in piece else ""); r.font.size = Pt(8); r.font.color.rgb = MUTED
            if " " not in piece:
                _field(r, piece)

    # ---- text
    def rich(self, p, text, size=None, bold=False):
        for tok in re.split(r"(\*\*.+?\*\*|\$.+?\$|\*[^*]+?\*)", text):
            if not tok:
                continue
            if tok.startswith("$"):
                p._p.append(parse_xml(etree.tostring(_xslt()(etree.fromstring(latex_to_mathml(tok[1:-1]).encode()))).decode()))
                continue
            if tok.startswith("**"):
                r = p.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("*"):
                r = p.add_run(tok[1:-1]); r.italic = True; r.bold = bold
            else:
                r = p.add_run(tok); r.bold = bold
            if size:
                r.font.size = Pt(size)
        return p

    def title(self, text, byline=None):
        p = self.doc.add_paragraph(); p.paragraph_format.space_after = Pt(8); p.paragraph_format.line_spacing = 1.05
        r = p.add_run(text); r.font.name = HEAD_FONT; r.font.size = Pt(24)
        if byline:
            q = self.doc.add_paragraph(); r = q.add_run(byline); r.italic = True
            self.rule()

    def rule(self):
        p = self.doc.add_paragraph(); p.paragraph_format.space_after = Pt(4)
        pb = OxmlElement("w:pBdr"); b = OxmlElement("w:bottom")
        for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", RULE)):
            b.set(qn(f"w:{k}"), v)
        pb.append(b); p._p.get_or_add_pPr().append(pb)

    def h1(self, text):
        p = self.doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text); r.font.name = HEAD_FONT; r.font.size = Pt(16)

    def h2(self, text):
        p = self.doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text); r.font.name = HEAD_FONT; r.font.size = Pt(12.5)

    def p(self, text, keep_next=False, size=None):
        para = self.doc.add_paragraph(); self.rich(para, text, size=size)
        para.paragraph_format.keep_with_next = keep_next
        return para

    def items(self, entries, numbered=False):
        for i, e in enumerate(entries, 1):
            para = self.doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(0.25); para.paragraph_format.first_line_indent = Inches(-0.2)
            para.paragraph_format.space_after = Pt(3)
            para.paragraph_format.tab_stops.add_tab_stop(Inches(0.25))
            self.rich(para, f"{i}.\t{e}" if numbered else f"•\t{e}")

    def equation(self, latex):
        para = self.doc.add_paragraph()
        node = parse_xml(etree.tostring(_xslt()(etree.fromstring(latex_to_mathml(latex, display="block").encode()))).decode())
        para._p.append(node)

    def figure(self, path, caption, width=TEXT_W):
        para = self.doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True; para.paragraph_format.space_before = Pt(4)
        para.add_run().add_picture(str(path), width=Inches(width))
        self.rich(self.doc.add_paragraph(), caption, size=9)

    def table(self, header, rows, widths, bold_rows=(), size=9.5, align=None, note=None):
        scale = min(1.0, TEXT_W / sum(widths))
        widths = [w * scale for w in widths]
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.LEFT
        t.autofit = False
        line = {"val": "single", "sz": 4, "space": 0, "color": "DAD8D2"}
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                c = t.cell(i, j); c.text = ""
                par = c.paragraphs[0]
                par.paragraph_format.space_after = Pt(1); par.paragraph_format.space_before = Pt(1)
                par.paragraph_format.line_spacing = 1.0
                par.paragraph_format.keep_with_next = i < len(rows) and len(rows) <= 12
                if align and align[j] == "r":
                    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                self.rich(par, str(val), size=size, bold=(i - 1 in bold_rows))
                if i == 0:
                    for r in par.runs:
                        r.font.color.rgb = MUTED
                    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), HEAD_FILL)
                    c._tc.get_or_add_tcPr().append(shd)
                set_cell_border(c, top=line, bottom=line, left=line, right=line)
            t.rows[i]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true")
        t.rows[0]._tr.get_or_add_trPr().append(th)
        fix_table_widths(t, widths)
        after = self.doc.add_paragraph(); after.paragraph_format.space_before = Pt(3); after.paragraph_format.space_after = Pt(8)
        if note:
            self.rich(after, note, size=8.5)

    def save(self, path):
        self.doc.save(str(path))
