import pymupdf

def openResumePdf(filename):
    doc = pymupdf.open(filename)
    return doc