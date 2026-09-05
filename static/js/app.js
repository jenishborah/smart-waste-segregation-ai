document.addEventListener("DOMContentLoaded", () => {

    const uploadBox = document.getElementById("uploadBox");
    const imageInput = document.getElementById("imageInput");
    const chooseButton = document.getElementById("chooseButton");

    const previewSection =
        document.getElementById("previewSection");

    const previewImage =
        document.getElementById("previewImage");

    const fileName =
        document.getElementById("fileName");

    const fileSize =
        document.getElementById("fileSize");

    const removeButton =
        document.getElementById("removeButton");

    const analyzeButton =
        document.getElementById("analyzeButton");

    const loading =
        document.getElementById("loading");

    const errorBox =
        document.getElementById("errorBox");

    const errorText =
        document.getElementById("errorText");

    const results =
        document.getElementById("results");

    const prediction =
        document.getElementById("prediction");

    const confidence =
        document.getElementById("confidence");

    const confidenceBar =
        document.getElementById("confidenceBar");

    const resultImage =
        document.getElementById("resultImage");

    const qualityStatus =
        document.getElementById("qualityStatus");

    const topPredictions =
        document.getElementById("topPredictions");

    const simpleExplanation =
        document.getElementById("simpleExplanation");

    const gradcamImage =
        document.getElementById("gradcamImage");

    const gradcamUnavailable =
        document.getElementById("gradcamUnavailable");

    const guidanceText =
        document.getElementById("guidanceText");

    const resetButton =
        document.getElementById("resetButton");


    let selectedFile = null;


    // --------------------------------------------------------
    // FILE PICKER
    // --------------------------------------------------------

    chooseButton.addEventListener("click", (event) => {
        event.stopPropagation();
        imageInput.click();
    });


    uploadBox.addEventListener("click", () => {

        if (!selectedFile) {
            imageInput.click();
        }

    });


    imageInput.addEventListener("change", (event) => {

        const file = event.target.files[0];

        if (file) {
            selectFile(file);
        }

    });


    // --------------------------------------------------------
    // DRAG AND DROP
    // --------------------------------------------------------

    uploadBox.addEventListener(
        "dragover",
        (event) => {

            event.preventDefault();

            uploadBox.classList.add("dragging");

        }
    );


    uploadBox.addEventListener(
        "dragleave",
        () => {

            uploadBox.classList.remove(
                "dragging"
            );

        }
    );


    uploadBox.addEventListener(
        "drop",
        (event) => {

            event.preventDefault();

            uploadBox.classList.remove(
                "dragging"
            );

            const file =
                event.dataTransfer.files[0];

            if (file) {
                selectFile(file);
            }

        }
    );


    // --------------------------------------------------------
    // SELECT FILE
    // --------------------------------------------------------

    function selectFile(file) {

        if (!file.type.startsWith("image/")) {

            showError(
                "Please select a JPG, PNG or WEBP image."
            );

            return;
        }


        if (file.size > 10 * 1024 * 1024) {

            showError(
                "The image is too large. Maximum size is 10 MB."
            );

            return;
        }


        selectedFile = file;

        hideError();

        hideResults();


        const reader = new FileReader();


        reader.onload = (event) => {

            previewImage.src =
                event.target.result;

            resultImage.src =
                event.target.result;

            fileName.textContent =
                file.name;

            fileSize.textContent =
                formatFileSize(file.size);

            previewSection.classList.remove(
                "hidden"
            );

            uploadBox.classList.add(
                "hidden"
            );

        };


        reader.onerror = () => {

            showError(
                "The selected image could not be previewed."
            );

        };


        reader.readAsDataURL(file);

    }


    // --------------------------------------------------------
    // REMOVE
    // --------------------------------------------------------

    removeButton.addEventListener(
        "click",
        (event) => {

            event.stopPropagation();

            resetEverything();

        }
    );


    // --------------------------------------------------------
    // ANALYZE
    // --------------------------------------------------------

    analyzeButton.addEventListener(
        "click",
        async () => {

            if (!selectedFile) {

                showError(
                    "Please select an image first."
                );

                return;
            }


            hideError();

            hideResults();

            previewSection.classList.add(
                "hidden"
            );

            loading.classList.remove(
                "hidden"
            );


            const formData =
                new FormData();

            formData.append(
                "image",
                selectedFile
            );


            try {

                const response =
                    await fetch(
                        "/analyze",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                let data;

                try {

                    data =
                        await response.json();

                } catch (jsonError) {

                    throw new Error(
                        "The server returned an invalid response."
                    );

                }


                if (!response.ok || !data.success) {

                    throw new Error(
                        data.error ||
                        "The image could not be analyzed."
                    );

                }


                displayResults(data);

            } catch (error) {

                console.error(
                    "Analysis error:",
                    error
                );

                showError(
                    error.message
                );

                previewSection.classList.remove(
                    "hidden"
                );

            } finally {

                loading.classList.add(
                    "hidden"
                );

            }

        }
    );


    // --------------------------------------------------------
    // DISPLAY RESULTS
    // --------------------------------------------------------

    function displayResults(data) {

        results.classList.remove(
            "hidden"
        );


        // Prediction

        prediction.textContent =
            data.prediction.class;


        confidence.textContent =
            data.prediction.confidence.toFixed(2)
            + "%";


        confidenceBar.style.width =
            data.prediction.confidence
            + "%";


        // Image quality

        qualityStatus.textContent =
            "Image quality: "
            + data.quality.status;


        // Actual uploaded image

        resultImage.src =
            data.image.url;


        // Top predictions

        topPredictions.innerHTML = "";


        data.top_predictions.forEach(
            (item) => {

                const row =
                    document.createElement("div");

                row.className =
                    "prediction-row";


                const name =
                    document.createElement("span");

                name.textContent =
                    item.class;


                const value =
                    document.createElement("strong");

                value.textContent =
                    item.confidence.toFixed(2)
                    + "%";


                row.appendChild(name);

                row.appendChild(value);

                topPredictions.appendChild(row);

            }
        );


        // Simple explanation

        simpleExplanation.textContent =
            data.why_this_result;


        // Grad-CAM

        gradcamImage.classList.add(
            "hidden"
        );

        gradcamUnavailable.classList.add(
            "hidden"
        );


        if (
            data.gradcam &&
            data.gradcam.available &&
            data.gradcam.url
        ) {

            gradcamImage.src =
                data.gradcam.url
                + "?t="
                + Date.now();


            gradcamImage.onload = () => {

                gradcamImage.classList.remove(
                    "hidden"
                );

            };


            gradcamImage.onerror = () => {

                gradcamImage.classList.add(
                    "hidden"
                );

                gradcamUnavailable.textContent =
                    "Grad-CAM image could not be displayed.";

                gradcamUnavailable.classList.remove(
                    "hidden"
                );

            };

        } else {

            gradcamUnavailable.textContent =
                data.gradcam &&
                data.gradcam.message
                    ? data.gradcam.message
                    : "Grad-CAM visualization is not available for this image.";

            gradcamUnavailable.classList.remove(
                "hidden"
            );

        }


        // Guidance

        guidanceText.textContent =
            getGuidance(
                data.prediction.class
            );


        // Scroll to result

        setTimeout(() => {

            results.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }, 100);

    }


    // --------------------------------------------------------
    // GUIDANCE
    // --------------------------------------------------------

    function getGuidance(category) {

        const value =
            category.toLowerCase();


        if (value.includes("plastic")) {

            return (
                "Empty and rinse the plastic item "
                + "when appropriate, then place it in "
                + "the designated plastic recycling stream. "
                + "Follow your local recycling rules."
            );

        }


        if (value.includes("paper")) {

            return (
                "Keep paper clean and dry where possible "
                + "and place it in the appropriate paper "
                + "recycling stream."
            );

        }


        if (value.includes("cardboard")) {

            return (
                "Flatten clean cardboard when practical "
                + "and place it in the appropriate cardboard "
                + "recycling stream."
            );

        }


        if (value.includes("glass")) {

            return (
                "Place glass in the designated glass "
                + "collection stream and follow local "
                + "handling rules."
            );

        }


        if (value.includes("metal")) {

            return (
                "Empty the metal item when appropriate "
                + "and place it in the designated metal "
                + "recycling stream."
            );

        }


        if (
            value.includes("organic") ||
            value.includes("food") ||
            value.includes("vegetation")
        ) {

            return (
                "Place organic material in the designated "
                + "organic or composting stream where "
                + "available."
            );

        }


        if (
            value.includes("battery") ||
            value.includes("hazardous")
        ) {

            return (
                "Do not place this with ordinary household "
                + "recycling. Use the appropriate hazardous "
                + "waste or special collection service."
            );

        }


        return (
            "Follow your local waste segregation rules "
            + "for this category."
        );

    }


    // --------------------------------------------------------
    // ERROR
    // --------------------------------------------------------

    function showError(message) {

        errorText.textContent =
            message;

        errorBox.classList.remove(
            "hidden"
        );

    }


    function hideError() {

        errorBox.classList.add(
            "hidden"
        );

        errorText.textContent = "";

    }


    // --------------------------------------------------------
    // RESULTS
    // --------------------------------------------------------

    function hideResults() {

        results.classList.add(
            "hidden"
        );

    }


    // --------------------------------------------------------
    // RESET
    // --------------------------------------------------------

    resetButton.addEventListener(
        "click",
        () => {

            resetEverything();

            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });

        }
    );


    function resetEverything() {

        selectedFile = null;

        imageInput.value = "";

        previewImage.src = "";

        resultImage.src = "";

        gradcamImage.src = "";

        previewSection.classList.add(
            "hidden"
        );

        results.classList.add(
            "hidden"
        );

        loading.classList.add(
            "hidden"
        );

        uploadBox.classList.remove(
            "hidden"
        );

        hideError();

        confidenceBar.style.width =
            "0%";

    }


    // --------------------------------------------------------
    // FILE SIZE
    // --------------------------------------------------------

    function formatFileSize(bytes) {

        if (bytes < 1024) {
            return bytes + " B";
        }

        if (bytes < 1024 * 1024) {
            return (
                (bytes / 1024).toFixed(1)
                + " KB"
            );
        }

        return (
            (bytes / (1024 * 1024)).toFixed(1)
            + " MB"
        );

    }

});