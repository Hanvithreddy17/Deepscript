# Frontend Directory

## Directory Purpose

This directory will store the client-side user interface application for DeepScript, allowing users to upload inscription images, trigger analysis, and view script identification results.

> **Status Notice:** The frontend user interface is **NOT implemented yet**. The workflow below details the planned user interaction model.

---

## Planned User Interface Flow

1. **Upload Inscription Image:** User selects or drops an inscription image into an upload area.
2. **Display Processing State:** Interface provides visual progress/loading indicators while analysis takes place.
3. **Send Image to Backend:** Image payload is transmitted asynchronously to the FastAPI backend `POST /predict` service.
4. **Display Predicted Script:** Interface renders the identified ancient Indian script label clearly.
5. **Display Model Score:** Renders the associated confidence or similarity score as percentage or metric indicator.
6. **Upload Another Image:** Provides seamless reset controls to perform subsequent script classifications.
