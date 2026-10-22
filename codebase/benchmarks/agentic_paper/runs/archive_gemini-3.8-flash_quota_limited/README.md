Episodes run on `gemini-3.8-flash` before the benchmark switched Gemini backbone.

Reason: the free tier of `gemini-3.8-flash` allows 20 generate_content requests per day per project
(429 QuotaFailure quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue 20, observed on all 14 valid keys
on 2026-10-21), far below what the benchmark needs. These partial results are kept for transparency, are NOT scored and are not
mixed with any reported number.
