document.addEventListener("DOMContentLoaded", function () {
    // File Upload Validation
    const fileInput = document.getElementById("file");
    if (fileInput) {
        fileInput.addEventListener("change", function () {
            const allowedExtensions = /(\.csv)$/i;
            if (!allowedExtensions.exec(fileInput.value)) {
                alert("Invalid file type. Please upload a CSV file.");
                fileInput.value = "";
            }
        });
    }

    // Flash Message Auto-hide
    const flashMessages = document.querySelectorAll(".flash");
    flashMessages.forEach((message) => {
        setTimeout(() => {
            message.style.display = "none";
        }, 3000);
    });

    // Form Validation for Login
    const loginForm = document.querySelector("form");
    if (loginForm) {
        loginForm.addEventListener("submit", function (event) {
            const username = document.getElementById("username").value.trim();
            const password = document.getElementById("password").value.trim();

            if (username === "" || password === "") {
                alert("Please enter both username and password.");
                event.preventDefault();
            }
        });
    }
});
