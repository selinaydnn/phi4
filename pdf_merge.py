import fitz


def merge_pdfs(pdf_list, output_pdf):
    merged_pdf = fitz.open()

    for pdf in pdf_list:
        with fitz.open(pdf) as doc:
            merged_pdf.insert_pdf(doc)

    merged_pdf.save(output_pdf)
    print(f"PDF dosyaları başarıyla birleştirildi: {output_pdf}")


# Kullanım
pdf_files = ["/Users/selinaydin/PycharmProjects/pythonProject7/source_doc/newmerge.pdf", "/Users/selinaydin/PycharmProjects/pythonProject7/data/webscrapinggüncel_data.xlsx - Sheet1.pdf", "/Users/selinaydin/PycharmProjects/pythonProject7/data/öğrenci-işleri.xlsx - Sayfa1.pdf"]
output_file = "source_doc/birlesik_dosya.pdf"

merge_pdfs(pdf_files, output_file)
