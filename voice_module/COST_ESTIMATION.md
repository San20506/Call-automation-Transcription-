# Industrial Scale Cost Estimation (Maximum Worst-Case Scenario)

This document provides a "Worst-Case" cost estimation for scaling to **10,000 hours per month**, assuming premium services and no spot-instance savings.

---

## 1. Speaker Diarization (Compute)
**Worst Case**: Using AWS On-Demand Instances (No Spot Savings).
*   **Instance**: `g4dn.xlarge` (Required for Pyannote GPU inference).
*   **Cost**: **$0.526 per hour** (AWS US East N. Virginia).
*   **Throughput**: 1 instance processes ~10 hours of audio per hour (Conservative estimate).
*   **Cost per Audio Hour**: $0.526 / 10 = **$0.053**

---

## 2. Transcription (Sarvam AI)
**Standard Rate**: Sarvam charges a fixed rate per second of audio.
*   **Service**: Speech to Text (Hindi).
*   **Rate**: ₹30 per hour ≈ **$0.36 per hour**.

---

## 3. Translation (Google Gemini LLM) - The "Premium" Component
**Worst Case**: Using **Gemini 1.5 Pro** (Most expensive, highest quality) with no caching.
*   **Pricing**:
    *   Input: $1.25 / 1M tokens
    *   Output: $5.00 / 1M tokens
*   **Estimation**:
    *   1 Hour Audio ≈ 9,000 words ≈ 12,000 tokens (Input).
    *   Translation Output ≈ 12,000 tokens (Output).
    *   Input Cost: (12,000 / 1,000,000) * $1.25 = $0.015
    *   Output Cost: (12,000 / 1,000,000) * $5.00 = $0.060
*   **Total Cost per Audio Hour**: **$0.075**
*   *Note: This is significantly cheaper than Google Translate API ($0.90/hr) while being "Premium" quality.*

---

## 🚨 TOTAL WORST-CASE COST (Per Hour of Audio)

| Component | Service | Worst-Case Cost |
| :--- | :--- | :--- |
| **Diarization** | AWS `g4dn.xlarge` (On-Demand) | **$0.053** |
| **Transcription** | Sarvam AI (Speech-to-Text) | **$0.360** |
| **Translation** | Google Gemini 1.5 Pro (LLM) | **$0.075** |
| **Storage** | S3 Standard + Data Transfer | **$0.012** |
| **Failover Buffer** | 10% Contingency | **$0.050** |
| **TOTAL** | | **~$0.55 / hour** |

### 📈 Monthly Projection (10,000 Hours)
*   **Total Monthly**: **$5,500**
*   **Per Minute**: **$0.009 (less than 1 cent)**

---

## 🔗 Pricing Reference Links (for Verification)
1.  **AWS EC2 Pricing**: [aws.amazon.com/ec2/pricing/on-demand/](https://aws.amazon.com/ec2/pricing/on-demand/) (Search `g4dn.xlarge`)
2.  **Sarvam AI Pricing**: [sarvam.ai/pricing](https://sarvam.ai/pricing) (Look for Speech-to-text ₹30/hour)
3.  **Google Gemini API**: [ai.google.dev/pricing](https://ai.google.dev/pricing) (Check "Gemini 1.5 Pro" Pay-as-you-go)

---

## 💡 Comparison: Why Gemini is better/cheaper?
*   **Google Translate API (Old way)**: Charges per character ($20/1M chars). 1 Hour ≈ 45k chars ≈ **$0.90**.
*   **Gemini 1.5 Pro (New way)**: Charges per token. 1 Hour ≈ 12k tokens ≈ **$0.075**.
*   **Result**: Gemini Pro is **12x cheaper** and provides **context-aware** translation.

