from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import PatientRecordForm
from .models import PatientRecord
from .web3 import w3

def store_data(request):
    if request.method == "POST":
        if w3.is_connected():
            return redirect('view_records')

@login_required(login_url='/login/')
def upload_record(request):
    if request.method == 'POST':
        form = PatientRecordForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('view_records')
    else:
        form = PatientRecordForm()
    return render(request, 'upload_record.html', {'form': form})

@login_required(login_url='/login/')
def view_records(request):
    records = PatientRecord.objects.all()
    return render(request, 'view_records.html', {'records': records})

@login_required(login_url='/login/')
def dashboard(request):
    return render(request, 'dashboard.html')
