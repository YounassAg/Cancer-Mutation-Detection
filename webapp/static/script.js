/**
 * Cancer Mutation Detection — Client-Side Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const btnPredict = document.getElementById('btn-predict');
    const resultsCard = document.getElementById('results-card');
    const emptyState = document.getElementById('result-empty');
    const resultsContent = document.getElementById('results-content');
    const errorBanner = document.getElementById('error-banner');
    const chipRow = document.getElementById('chip-row');

    // Load example variants on startup
    fetch('/example-variants')
        .then(res => res.json())
        .then(variants => {
            variants.forEach(v => {
                const chip = document.createElement('div');
                chip.className = 'chip';
                chip.innerHTML = `<div class="dot"></div>${v.label}`;
                chip.title = v.description;
                chip.addEventListener('click', () => fillForm(v));
                chipRow.appendChild(chip);
            });
        })
        .catch(err => console.error('Failed to load examples:', err));

    // Fill form with selected variant
    function fillForm(v) {
        document.getElementById('gene_symbol').value = v.GeneSymbol;
        document.getElementById('gene_id').value = v.GeneID;
        document.getElementById('position').value = v.PositionVCF;
        document.getElementById('ref_allele').value = v.ReferenceAlleleVCF;
        document.getElementById('alt_allele').value = v.AlternateAlleleVCF;
        document.getElementById('var_type').value = v.Type;
        document.getElementById('chromosome').value = v.Chromosome;
        document.getElementById('origin').value = v.OriginSimple;
        
        // Highlight form to indicate it was auto-filled
        form.querySelectorAll('input, select').forEach(el => {
            el.style.backgroundColor = '#F0FDFA';
            setTimeout(() => el.style.backgroundColor = '', 500);
        });
    }

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Reset UI state
        errorBanner.classList.remove('visible');
        btnPredict.classList.add('loading');
        btnPredict.disabled = true;
        
        // Gather data
        const payload = {
            GeneSymbol: document.getElementById('gene_symbol').value,
            GeneID: document.getElementById('gene_id').value,
            PositionVCF: document.getElementById('position').value,
            ReferenceAlleleVCF: document.getElementById('ref_allele').value,
            AlternateAlleleVCF: document.getElementById('alt_allele').value,
            Type: document.getElementById('var_type').value,
            Chromosome: document.getElementById('chromosome').value,
            OriginSimple: document.getElementById('origin').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            renderResults(data);
        } catch (error) {
            errorBanner.textContent = error.message;
            errorBanner.classList.add('visible');
            resultsCard.classList.remove('visible');
        } finally {
            btnPredict.classList.remove('loading');
            btnPredict.disabled = false;
        }
    });

    // Render the results into the DOM
    function renderResults(data) {
        // Hide empty state, show results container
        emptyState.style.display = 'none';
        resultsContent.style.display = 'block';
        
        // Determine theme based on oncogenicity
        const theme = data.is_oncogenic ? 'oncogenic' : 'benign';
        const iconAlert = data.is_oncogenic 
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>'
            : '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>';
            
        // Build the HTML structure
        let html = `
            <div class="classification-banner ${theme}">
                <div class="classification-label">Final Classification</div>
                <div class="classification-text">${data.classification}</div>
            </div>
            
            <div class="gauge-container">
                <svg class="gauge-svg" width="120" height="120" viewBox="0 0 120 120">
                    <circle class="gauge-bg" cx="60" cy="60" r="50"></circle>
                    <circle class="gauge-fill" cx="60" cy="60" r="50" stroke="${data.is_oncogenic ? 'var(--danger)' : 'var(--success)'}" stroke-dasharray="314" stroke-dashoffset="314" id="gauge-circle"></circle>
                    <g class="gauge-text-group">
                        <text class="gauge-value" x="60" y="56">${data.confidence}%</text>
                        <text class="gauge-subtext" x="60" y="74">Confidence</text>
                    </g>
                </svg>
            </div>
            
            <div class="risk-bar-container">
                <div class="risk-bar-label">
                    <span>Risk Level</span>
                    <span>${data.risk_level}</span>
                </div>
                <div class="risk-bar">
                    <div class="risk-bar-fill ${data.risk_level.toLowerCase()}"></div>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-section-title">Biological Context</div>
                <div class="detail-row">
                    <span class="detail-key">Mutation Type</span>
                    <span class="detail-val">${data.mutation_type}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-key">Variant</span>
                    <span class="detail-val">${data.variant.gene_display}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-key">DNA Change</span>
                    <span class="detail-val">${data.variant.dna_change}</span>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-section-title">Clinical Advice</div>
                <ul class="advice-list">
                    ${data.advice.map((msg, i) => `
                        <li class="advice-item ${theme}" style="animation-delay: ${i * 0.1}s">
                            <span class="advice-icon">${iconAlert}</span>
                            <span>${msg}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
        
        resultsContent.innerHTML = html;
        
        // Show the card
        resultsCard.classList.add('visible');
        
        // Animate the gauge
        setTimeout(() => {
            const circle = document.getElementById('gauge-circle');
            if (circle) {
                const circumference = 2 * Math.PI * 50; // ~314
                const offset = circumference - (data.confidence / 100) * circumference;
                circle.style.strokeDashoffset = offset;
            }
        }, 50);
    }
});
