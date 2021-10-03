function updateFilename() {
    lbl_filename = document.getElementById("xray-name");
    btn_submit = document.getElementById("submit");
    file = document.getElementById("xray");

    lbl_filename.innerHTML = file.files[0].name;
    btn_submit.disabled = false;
}
