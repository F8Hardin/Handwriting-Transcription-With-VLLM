import sys
import xml.etree.ElementTree as ET

NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}"
REGION = NS + "AttrRegion"

BLOCK, LINE, WORD = "3", "2", "1"
HANDWRITTEN = "2"

#Returns the text from the CVL XML file
def extract_text_cvl(xml_path: str) -> str:
    raw = open(xml_path, "rb").read().decode("cp1252").replace('encoding="UTF-16"', "")
    root = ET.fromstring(raw)

    for block in root.iter(REGION):
        if block.get("attrType") == BLOCK and block.get("fontType") == HANDWRITTEN:
            lines = []
            for line in block.findall(REGION):
                if line.get("attrType") == LINE:
                    words = [w.get("text", "") for w in line.iter(REGION) if w.get("attrType") == WORD]
                    lines.append(" ".join(words))
            return "\n".join(lines)
    return ""

#Returns the text from the SMHD txt file
def extract_text_smhd(txt_path: str) -> str:
    with open(txt_path, "r", encoding='cp1252') as f:
        content = f.read()
    return content
