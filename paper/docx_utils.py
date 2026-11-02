"""python-docx helpers for an IEEE conference-style paper (IEEEtran / IEEE Word template conventions).

Layout: US Letter, top 0.75", bottom 1", left/right 0.625", two 3.5" columns with a 0.25" gap.
Title 24 pt; author block 11/9 pt; abstract and index terms 9 pt bold; body 10 pt Times New Roman;
section headings small caps with Roman numerals; subsections italic "A."; third level run-in
italic "1)"; figure captions 8 pt below ("Fig. 1."); table captions 8 pt above ("TABLE I" + small
caps title); references 8 pt. Floats wider than a column span both columns (like figure*).
Equations are native Word (OMML) display equations, numbered flush right.
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, Emu
from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree

XSL_CANDIDATES = [Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
                  Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL")]
_XSLT = None
FONT = "Times New Roman"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
         "XVI", "XVII", "XVIII", "XIX", "XX"]

# IEEE conference template, A4: top 19 mm, bottom 43 mm, left/right 14.32 mm,
# two 88.9 mm columns with a 4.22 mm gap (all in inches below)
MM = 1 / 25.4
PAGE_W, PAGE_H = 210 * MM, 297 * MM
M_TOP, M_BOTTOM, M_SIDE = 19 * MM, 43 * MM, 14.32 * MM
COL_W, COL_GAP = 88.9 * MM, 4.22 * MM
FULL_W = PAGE_W - 2 * M_SIDE           # 181.36 mm = 7.14"


def fix_table_widths(table, widths_in):
    """Write the column widths into tblGrid, tblW and every cell, with a fixed layout.
    Word honours per-cell widths, but LibreOffice / Google Docs / WPS lay a table out from its grid;
    python-docx writes the grid at full page width, which made narrow tables spill out of a column."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    tw = int(round(sum(widths_in) * 1440))
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
    tblW.set(qn("w:type"), "dxa"); tblW.set(qn("w:w"), str(tw))
    lay = tblPr.find(qn("w:tblLayout"))
    if lay is None:
        lay = OxmlElement("w:tblLayout"); tblPr.append(lay)
    lay.set(qn("w:type"), "fixed")
    ind = tblPr.find(qn("w:tblInd"))
    if ind is None:
        ind = OxmlElement("w:tblInd"); tblPr.append(ind)
    ind.set(qn("w:w"), "0"); ind.set(qn("w:type"), "dxa")
    for gc, w in zip(tbl.tblGrid.findall(qn("w:gridCol")), widths_in):
        gc.set(qn("w:w"), str(int(round(w * 1440))))
    for row in table.rows:
        for c, w in zip(row.cells, widths_in):
            c.width = Inches(w)
BODY_PT, SMALL_PT, ABSTRACT_PT = 10, 8, 9


def _xslt():
    global _XSLT
    if _XSLT is None:
        for p in XSL_CANDIDATES:
            if p.exists():
                _XSLT = etree.XSLT(etree.parse(str(p)))
                break
    return _XSLT


def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders"); tcPr.append(borders)
    for edge, val in kw.items():
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}"); borders.append(el)
        for k, v in val.items():
            el.set(qn(f"w:{k}"), str(v))


def _set_cols(section, n):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols"); sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(COL_GAP * 1440)))
    cols.set(qn("w:equalWidth"), "1")


def _geometry(section):
    section.page_width, section.page_height = Inches(PAGE_W), Inches(PAGE_H)
    section.top_margin, section.bottom_margin = Inches(M_TOP), Inches(M_BOTTOM)
    section.left_margin = section.right_margin = Inches(M_SIDE)
    section.header_distance = Inches(0.5); section.footer_distance = Inches(0.5)


class Paper:
    def __init__(self):
        self.doc = Document()
        self.fig_n = self.tab_n = self.eq_n = self.sec_n = self.sub_n = self.subsub_n = 0
        self.labels = {}
        self.ncols = 1
        _geometry(self.doc.sections[0])
        _set_cols(self.doc.sections[0], 1)
        st = self.doc.styles["Normal"]
        st.font.name = FONT; st.font.size = Pt(BODY_PT)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = st.paragraph_format
        pf.space_after = Pt(0); pf.space_before = Pt(0); pf.line_spacing = 1.0
        # IEEE conference papers carry no page numbers (added by the publisher)

    # ------------------------------------------------------------ column handling
    @property
    def cur_width(self):
        return COL_W if self.ncols == 2 else FULL_W

    def columns(self, n):
        """Start a continuous section with n columns (Word balances the columns above the break)."""
        if n == self.ncols:
            return
        sec = self.doc.add_section(WD_SECTION.CONTINUOUS)
        _geometry(sec)
        _set_cols(sec, n)
        self.ncols = n

    # ------------------------------------------------------------ text
    def para(self, text="", align="justify", size=None, bold=False, italic=False, indent=True, space_after=0,
             keep_next=False):
        p = self.doc.add_paragraph()
        p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                       "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        if indent and align == "justify":
            p.paragraph_format.first_line_indent = Inches(0.14)
        p.paragraph_format.space_after = Pt(space_after)
        if keep_next:
            p.paragraph_format.keep_with_next = True
        if text:
            self.rich(p, text, size=size, bold=bold, italic=italic)
        return p

    def rich(self, p, text, size=None, bold=False, italic=False):
        """Minimal inline markup: **bold**, *italic*, $latex$ (inline equation)."""
        import re
        for tok in re.split(r"(\*\*.+?\*\*|\$.+?\$|\*[^*]+?\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = p.add_run(tok[2:-2]); r.bold = True; r.italic = italic
            elif tok.startswith("$"):
                self._omath(p, tok[1:-1]); continue
            elif tok.startswith("*"):
                r = p.add_run(tok[1:-1]); r.italic = not italic; r.bold = bold
            else:   # typographic minus for negative numbers (not for dates, ranges or hyphenated words)
                tok = re.sub(r"(^|[\s(\[=/:])-(?=\d)", "\\1\u2212", tok)
                r = p.add_run(tok); r.bold = bold; r.italic = italic
            if size:
                r.font.size = Pt(size)
        return p

    def _omath(self, p, latex):
        mml = latex_to_mathml(latex)
        omml = _xslt()(etree.fromstring(mml.encode()))
        p._p.append(parse_xml(etree.tostring(omml).decode()))

    def heading(self, title):
        """IEEE level-1 heading: Roman numeral, small caps, centred."""
        self.sec_n += 1
        self.sub_n = 0
        self.subsub_n = 0
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(f"{ROMAN[self.sec_n - 1]}. {title}")
        r.font.size = Pt(BODY_PT); r.font.small_caps = True
        return p

    def heading_unnumbered(self, title):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title); r.font.size = Pt(BODY_PT); r.font.small_caps = True
        return p

    def subheading(self, title):
        """IEEE level-2 heading: 'A. Title' in italics, flush left."""
        self.sub_n += 1
        self.subsub_n = 0
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(f"{chr(64 + self.sub_n)}. {title}"); r.italic = True
        return p

    def subsubheading(self, title, text=""):
        """IEEE level-3 heading: run-in '1) Title:' in italics, followed by the paragraph text."""
        self.subsub_n += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Inches(0.14)
        r = p.add_run(f"{self.subsub_n}) {title}: "); r.italic = True
        if text:
            self.rich(p, text)
        return p

    # ------------------------------------------------------------ equations
    def equation(self, latex, label=None):
        """Numbered display equation: borderless 1x3 table in the current column, number flush right."""
        self.eq_n += 1
        if label:
            self.labels[label] = self.eq_n
        latex = latex.replace(r"\displaystyle", "").strip()
        w_total = self.cur_width
        t = self.doc.add_table(rows=1, cols=3)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        widths = [0.05 * w_total, 0.82 * w_total, 0.13 * w_total]
        fix_table_widths(t, widths)
        for j, w in enumerate(widths):
            c = t.cell(0, j)
            for e in ("top", "bottom", "left", "right"):
                set_cell_border(c, **{e: {"val": "nil"}})
            par = c.paragraphs[0]
            par.paragraph_format.space_before = Pt(3); par.paragraph_format.space_after = Pt(3)
            par.paragraph_format.first_line_indent = Inches(0)
        mid = t.cell(0, 1).paragraphs[0]
        mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
        omml = _xslt()(etree.fromstring(latex_to_mathml(latex, display="block").encode()))
        node = parse_xml(etree.tostring(omml).decode())
        if not node.tag.endswith("oMathPara"):
            para = OxmlElement("m:oMathPara"); para.append(node); node = para
        mid._p.append(node)
        rp = t.cell(0, 2).paragraphs[0]
        rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        rp.add_run(f"({self.eq_n})")
        t.cell(0, 2).vertical_alignment = 1
        return self.eq_n

    # ------------------------------------------------------------ figures & tables
    def figure(self, path, caption, width_in=None, label=None, wide=True):
        """Figure with an 8-pt 'Fig. n.' caption below. wide=True spans both columns (figure*)."""
        self.fig_n += 1
        if label:
            self.labels[label] = self.fig_n
        back = self.ncols
        if wide and self.ncols == 2:
            self.columns(1)
        w = width_in or (7.0 if self.ncols == 1 else 3.45)
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.add_run().add_picture(str(path), width=Inches(w))
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.space_before = Pt(3); c.paragraph_format.space_after = Pt(8)
        r = c.add_run(f"Fig. {self.fig_n}. "); r.font.size = Pt(SMALL_PT)
        self.rich(c, caption, size=SMALL_PT)
        if back != self.ncols:
            self.columns(back)
        return self.fig_n

    def table(self, header, rows, caption, col_widths=None, label=None, font_size=SMALL_PT, align=None, notes=None,
              wide=None):
        """IEEE table: 'TABLE I' and a small-caps title above, 8-pt text, rules at top, below the header and bottom.
        wide=None decides automatically: the table spans both columns if its columns do not fit 3.5"."""
        self.tab_n += 1
        if label:
            self.labels[label] = self.tab_n
        if wide is None:
            wide = (sum(col_widths) > COL_W) if col_widths else len(header) > 4
        back = self.ncols
        if wide and self.ncols == 2:
            self.columns(1)
        cp = self.doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(8); cp.paragraph_format.keep_with_next = True
        r = cp.add_run(f"TABLE {ROMAN[self.tab_n - 1]}"); r.font.size = Pt(SMALL_PT)
        cp2 = self.doc.add_paragraph()
        cp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp2.paragraph_format.space_after = Pt(3); cp2.paragraph_format.keep_with_next = True
        r = cp2.add_run(caption); r.font.size = Pt(SMALL_PT); r.font.small_caps = True
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        line = {"val": "single", "sz": 8, "space": 0, "color": "000000"}
        thin = {"val": "single", "sz": 4, "space": 0, "color": "000000"}
        none = {"val": "nil"}
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.text = ""
                par = cell.paragraphs[0]
                par.paragraph_format.space_after = Pt(0); par.paragraph_format.space_before = Pt(0)
                par.paragraph_format.first_line_indent = Inches(0)
                a = (align[j] if align else ("left" if j == 0 else "center"))
                par.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                                 "right": WD_ALIGN_PARAGRAPH.RIGHT}[a]
                self.rich(par, str(val) if str(val) != "" else "\u2014", size=font_size, bold=(i == 0))
                b = {"left": none, "right": none, "insideH": none, "insideV": none}
                b["top"] = line if i == 0 else (thin if i == 1 else none)
                b["bottom"] = line if i == len(rows) else none
                set_cell_border(cell, **b)
                par.paragraph_format.keep_with_next = (i < len(rows)) if len(rows) <= 12 else (i == 0)
            trPr = t.rows[i]._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
        th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true")
        t.rows[0]._tr.get_or_add_trPr().append(th)
        if not col_widths:
            avail = (COL_W if self.ncols == 2 else FULL_W) - 0.05
            col_widths = [avail / len(header)] * len(header)
        fix_table_widths(t, col_widths)
        n = self.doc.add_paragraph()
        n.paragraph_format.space_before = Pt(2); n.paragraph_format.space_after = Pt(8)
        if notes:
            self.rich(n, notes, size=7)
        if back != self.ncols:
            self.columns(back)
        return self.tab_n

    def save(self, path):
        self.doc.save(str(path))
