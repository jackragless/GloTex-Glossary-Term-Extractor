from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect
from . import pdf_to_text, glogen
import json


def baseRedirect(request):
    return redirect(text)

def document(request):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('document_upload')
        files_dict = {}
        if len(uploaded_files)>0:
            for file in uploaded_files:
                if file.name.endswith('.pdf'):
                    files_dict[file.name] = pdf_to_text.extract(file)
                elif file.name.endswith('.txt'):
                    files_dict[file.name] = file.read()
        # print(files_dict)
        request.session['download'] = json.dumps(glogen.glogenIterator(files_dict))
        # request.session['download'] = json.dumps(files_dict)
        request.session.save()
        return redirect('../download')

    return render(request, 'document.html')

def text(request):
    glogen_dict = None
    textbox_content = ''
    if request.method == "POST":
        textbox_content = str(request.POST.get('textbox'))
        if textbox_content:
            glogen_dict = glogen.glogen(textbox_content)

    return render(request, 'text.html', {'glogen_dict':glogen_dict, 'textbox_content':textbox_content})

def ack(request):
    return render(request, 'ack.html')

def download_file(request):
    filename = 'output.json'
    response = HttpResponse(request.session['download'], content_type="application/json")
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response