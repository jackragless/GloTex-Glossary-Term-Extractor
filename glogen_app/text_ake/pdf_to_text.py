import PyPDF2

def extract(pdffileobj):
	pdfreader=PyPDF2.PdfFileReader(pdffileobj)
	text = ""
	for pagenum in range(1,pdfreader.numPages):
		text+=pdfreader.getPage(pagenum).extractText()
	return text