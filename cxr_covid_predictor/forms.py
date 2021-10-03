from django import forms

class CXRUpload(forms.Form):
    file = forms.FileField()