function RunImageClassification() {
    const fileInput = document.getElementById('imageToAnalyze');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    fetch("/predictImage", {
        method: "POST",
        body: formData
    })
    .then(resp => resp.json())
    .then(data => {
        document.getElementById("system_response").innerText = JSON.stringify(data, null, 2);
    })
    .catch(err => {
        document.getElementById("system_response").innerText = "Error: " + err;
    });
}