document.addEventListener("DOMContentLoaded", () => {
    // DOM elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("resume-file");
    const removeFileBtn = document.getElementById("remove-file-btn");
    const fileInfo = document.getElementById("file-info");
    const fileNameText = document.getElementById("file-name-text");
    const dropzonePrompt = dropzone.querySelector(".dropzone-prompt");
    
    const form = document.getElementById("analysis-form");
    const resumeRawText = document.getElementById("resume-raw-text");
    const jdTitle = document.getElementById("jd-title");
    const jdRequirements = document.getElementById("jd-requirements");
    
    const resultsPanel = document.getElementById("results-panel");
    const emptyState = document.getElementById("empty-state");
    const loaderOverlay = document.getElementById("loader-overlay");
    const loaderText = loaderOverlay.querySelector(".loader-text");
    const reportContent = document.getElementById("report-content");
    
    // Metrics Elements
    const atsScoreText = document.getElementById("ats-score-text");
    const atsGaugeFill = document.getElementById("ats-gauge-fill");
    const matchPercentText = document.getElementById("match-percent-text");
    const matchFill = document.getElementById("match-fill");
    const skillsCoverageText = document.getElementById("skills-coverage-text");
    const skillsFill = document.getElementById("skills-fill");
    
    // Detailed Content Elements
    const reasoningSummaryText = document.getElementById("reasoning-summary-text");
    const strengthsList = document.getElementById("strengths-list");
    const weaknessesList = document.getElementById("weaknesses-list");
    const confidenceValueText = document.getElementById("confidence-value-text");
    const missingSkillsContainer = document.getElementById("missing-skills-container");
    const recruiterNotesText = document.getElementById("recruiter-notes-text");

    let uploadedFile = null;

    // --- Drag & Drop Event Listeners ---
    
    dropzone.addEventListener("click", () => {
        if (!uploadedFile) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    removeFileBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // Avoid triggering dropzone click
        resetFileUpload();
    });

    function handleFileSelect(file) {
        uploadedFile = file;
        fileNameText.textContent = file.name;
        dropzonePrompt.style.display = "none";
        fileInfo.style.display = "inline-flex";
    }

    function resetFileUpload() {
        uploadedFile = null;
        fileInput.value = "";
        fileInfo.style.display = "none";
        dropzonePrompt.style.display = "block";
    }

    // --- Tab Interactivity ---
    
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(btn.getAttribute("data-tab"));
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });

    // --- Form Submissions and API flow ---

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Show loader
        emptyState.style.display = "none";
        reportContent.style.display = "none";
        loaderOverlay.style.display = "flex";
        
        try {
            let extractedResumeText = resumeRawText.value.trim();

            // 1. If PDF uploaded, hit parse endpoint first
            if (uploadedFile) {
                loaderText.textContent = "Extracting document layout and parsing PDF...";
                
                const parseData = new FormData();
                parseData.append("file", uploadedFile);
                
                const parseResponse = await fetch("/api/parse", {
                    method: "POST",
                    body: parseData
                });
                
                if (!parseResponse.ok) {
                    throw new Error("Document parsing failed. Ensure file is not corrupt.");
                }
                
                const parseResult = await parseResponse.json();
                // Merge parsed resume representation back to text content for analytical parsing
                extractedResumeText = JSON.stringify(parseResult);
            }

            if (!extractedResumeText) {
                throw new Error("Please upload a resume file or paste raw resume text to analyze.");
            }

            // 2. Hit analyze endpoint
            loaderText.textContent = "Executing Qwen3-8B scoring checks...";
            
            const analyzeData = new FormData();
            analyzeData.append("resume_text", extractedResumeText);
            analyzeData.append("jd_title", jdTitle.value.trim());
            analyzeData.append("jd_requirements", jdRequirements.value.trim());
            
            const analyzeResponse = await fetch("/api/analyze", {
                method: "POST",
                body: analyzeData
            });

            if (!analyzeResponse.ok) {
                throw new Error("Matching analysis failed. Verify input syntax.");
            }

            const report = await analyzeResponse.json();
            renderReport(report);
            
        } catch (error) {
            alert(error.message);
            emptyState.style.display = "block";
            reportContent.style.display = "none";
        } finally {
            loaderOverlay.style.display = "none";
        }
    });

    // --- Rendering Report Data ---

    function renderReport(report) {
        // Animate metrics progress
        const score = report.ats_score;
        
        // 1. SVG Radial Fill Circle Animation
        // circumference of circle r=42 is 2*pi*42 = 263.89
        const strokeDashOffset = 263.8 - (263.8 * score / 100);
        atsGaugeFill.style.strokeDashoffset = strokeDashOffset;
        
        // Counter score animation
        animateCounter(atsScoreText, score);
        
        // Progress bars
        const matchPct = Math.round(report.job_match * 100);
        matchPercentText.textContent = `${matchPct}%`;
        matchFill.style.width = `${matchPct}%`;
        
        skillsCoverageText.textContent = `${report.skill_score}%`;
        skillsFill.style.width = `${report.skill_score}%`;
        
        // Detailed panels
        reasoningSummaryText.textContent = report.reasoning_summary;
        confidenceValueText.textContent = report.confidence_score ? report.confidence_score.toFixed(2) : "0.90";
        recruiterNotesText.textContent = `Matching calculations complete. Model identified alignment with skill clusters. ${report.reasoning_summary}`;
        
        // Populate strengths list
        strengthsList.innerHTML = "";
        report.strengths.forEach(str => {
            const li = document.createElement("li");
            li.textContent = str;
            strengthsList.appendChild(li);
        });
        
        // Populate weaknesses/gaps list
        weaknessesList.innerHTML = "";
        report.weaknesses.forEach(weak => {
            const li = document.createElement("li");
            li.textContent = weak;
            weaknessesList.appendChild(li);
        });
        
        // Populate missing skills badges
        missingSkillsContainer.innerHTML = "";
        if (report.missing_skills.length === 0) {
            const span = document.createElement("span");
            span.className = "skill-badge";
            span.style.borderColor = "var(--color-success)";
            span.style.color = "var(--color-success)";
            span.textContent = "All requirements matched!";
            missingSkillsContainer.appendChild(span);
        } else {
            report.missing_skills.forEach(skill => {
                const span = document.createElement("span");
                span.className = "skill-badge missing";
                span.textContent = skill;
                missingSkillsContainer.appendChild(span);
            });
        }
        
        // Display report panel
        reportContent.style.display = "flex";
    }

    function animateCounter(element, targetValue) {
        let start = 0;
        const duration = 1200; // ms
        const startTime = performance.now();
        
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function outQuad
            const easedProgress = progress * (2 - progress);
            const val = Math.floor(easedProgress * targetValue);
            
            element.textContent = val;
            
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = targetValue;
            }
        }
        requestAnimationFrame(update);
    }
});
