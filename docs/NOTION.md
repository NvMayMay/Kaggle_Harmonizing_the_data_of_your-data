# SDRF Agentic Extraction System -- Strategy & Roadmap

## Vision

Build a tool-augmented agentic system that achieves 0.75+ F1 on automated SDRF metadata extraction from proteomics papers. The core innovation is separating _what to extract_ (model reasoning) from _how to format_ (deterministic tool lookups), eliminating the ontology hallucination problem that limits all LLM-only approaches.

---

## Strategic Context

### The Problem with LLM-Only Approaches

After 15 iterations of pipeline development across the Kaggle competition, a clear pattern emerged:

| Approach | Score | Failure Mode |
|----------|-------|-------------|
| Single LLM extraction | 0.239 | Format hallucination, missed fields |
| Multi-round 3-model deliberation | 0.318 | Models amplify each other's format errors |
| Template matching from gold corpus | 0.144 | Similar instruments =/= similar experiments |
| Smart defaults + PRIDE + single LLM | 0.402 | LLM still guesses accession codes wrong ~40% |

The root cause is consistent: LLMs are asked to produce exact ontology strings (`NT=Q Exactive HF;AC=MS:1002523`) from memory. They get the concept right but the format wrong. Meanwhile, smart defaults alone beat multi-model deliberation because they never produce _wrong_ formats.

### The Solution: Tool-Augmented Extraction

The model's job becomes:
1. **Read the paper** and understand what experiment was done
2. **Decide what to look up** (which instrument, which modifications)
3. **Call tools** that return guaranteed-correct format strings
4. **Assemble the output** using tool results + biology extraction

This separates reasoning (model) from formatting (tools) -- each component does what it's best at.

---

## Architecture Summary

```
Paper Text + PXD
       |
       v
  Fine-tuned Qwen 2.5-7B
       |
       |-- pride_lookup(pxd)              --> instruments, organisms, mods, label type
       |-- ms_ontology_lookup(instrument) --> NT=...;AC=MS:XXXXX
       |-- unimod_lookup(modification)    --> NT=...;AC=UNIMOD:XX;TA=...;MT=...
       |-- sdrf_format_reference(column)  --> gold-standard format with frequency
       |-- paper_fetch(pxd)              --> full text from EuropePMC
       |
       v
  Final JSON Output
       |
       v
  Row Builder (deterministic)
       |
       v
  submission.csv
```

---

## Current Status

### DONE

- [x] **Ontology databases** -- PSI-MS OBO (600 terms), UNIMOD XML (1,560 mods), gold SDRF index (5,965 values from 281 PXDs)
- [x] **MCP server** -- 5 tools implemented and tested (pride_lookup, ms_ontology_lookup, unimod_lookup, paper_fetch, sdrf_format_reference)
- [x] **Training data** -- 103 tool-calling traces from gold SDRFs with real tool responses (1,391 messages, 541 tool calls)
- [x] **Inference pipeline** -- Agentic loop with GPT-4.1/RunPod/local model support, protected defaults, submission builder
- [x] **Deployment config** -- Dockerfile, requirements.txt for MCP server containerisation

### TO DO

- [ ] **Fine-tune Qwen 2.5-7B-Instruct** on tool-calling traces
  - LoRA r=32, alpha=64, 3-5 epochs
  - RunPod 1x H100, estimated 1-2 hours
  - Use existing training infrastructure (adapt from ICH Mixtral training)

- [ ] **Deploy fine-tuned model** on RunPod Serverless
  - Merge LoRA weights into base model
  - SGLang serving with tool-calling support
  - Verify tool-calling works end-to-end

- [ ] **Run end-to-end validation** on training PXDs
  - Leave-one-out: exclude target PXD from gold_sdrf.db during format reference lookups
  - Score against gold using competition scoring metric
  - Gate: F1 > 0.60 before submitting to Kaggle

- [ ] **Expand training data** (optional, for higher scores)
  - Fetch papers for 271 bigbio PXDs via EuropePMC
  - Generate additional tool-calling traces (up to 374 total)
  - Data augmentation: rephrase papers, vary tool call order

- [ ] **Deploy MCP server** to Azure Container App
  - Same pattern as existing CosmosDB MCP Toolkit deployment
  - API key auth (simple, no EasyAuth complexity)

---

## Key Decisions & Trade-offs

### Why Qwen 2.5-7B over larger models?

| Factor | Qwen 2.5-7B | GPT-4.1 / larger |
|--------|------------|-------------------|
| Tool calling | Native, first-class support | Works but black-box |
| Inference cost | ~$1 for 15 PXDs (RunPod) | ~$3-5 (API) |
| Fine-tuning | Full control, LoRA on 1 GPU | Not possible |
| Latency | ~5-10s per PXD | ~15-30s per PXD |
| Format learning | Trained on exact gold formats | Must be prompted |
| "not available" calibration | Learned from 374 examples | Must be instructed |

The model doesn't need to be large because the tools handle the hard part (ontology formatting). The model only needs to read papers accurately and know when to call which tool.

### Why not just use GPT-4.1 with tools (no fine-tuning)?

The inference pipeline supports GPT-4.1 as a fallback. The key advantage of fine-tuning:

1. **"not available" calibration** -- The model learns _from 374 examples_ when a field should be "not available" vs "Not Applicable". GPT-4.1 must be told via prompting and gets it wrong ~30% of the time.
2. **Tool calling patterns** -- The model learns the exact workflow: always call PRIDE first, then ontology lookup for each instrument, then UNIMOD for each modification. GPT-4.1 sometimes skips steps or calls tools unnecessarily.
3. **Output consistency** -- Fine-tuned model always outputs the same JSON schema. GPT-4.1 occasionally adds extra fields or changes key names.

### Why MCP over direct function calls?

The MCP server can be:
- Shared across multiple model deployments
- Updated independently (new ontology version = rebuild databases, no model changes)
- Tested in isolation
- Deployed as a microservice
- Reused for other proteomics tools

For development, the tools are called directly as Python functions (no MCP transport overhead). MCP transport is only needed for remote deployment.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 103 training examples insufficient | Model doesn't generalise | Medium | Data augmentation; GPT-4.1 fallback |
| Tool-calling format doesn't transfer | Model can't call tools at inference | Low | Use Qwen's native format; validated in similar projects |
| PRIDE API rate limiting | Can't fetch metadata for all PXDs | Medium | Cache responses; 2s delay between calls |
| Gold SDRF format reference leaks test answers | Overfitting to gold corpus | None | Test PXDs are confirmed absent from both training and bigbio |
| Fine-tuned model overfits to training PXDs | High train F1, low test F1 | Medium | Leave-one-out validation; regularisation |

---

## Metrics & Success Criteria

| Milestone | Target | Metric |
|-----------|--------|--------|
| Tools working | All 5 tools pass test suite | test_tools.py passes |
| Training data quality | Traces match gold output | Manual inspection of 10 examples |
| Fine-tuned model | Loss < 1.0 after 3 epochs | Training loss curve |
| Local validation | F1 > 0.60 on training PXDs | scoring.py evaluation |
| Kaggle submission | F1 > 0.50 on leaderboard | Competition score |
| Stretch goal | F1 > 0.70 on leaderboard | Competition score |

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Ontology databases + MCP server | 1 day | **Done** |
| Training data generation | 0.5 days | **Done** |
| Inference pipeline | 0.5 days | **Done** |
| Fine-tuning | 2-3 hours (compute) | Pending |
| Validation + iteration | 1 day | Pending |
| Submission | 1 hour | Pending |

**Total estimated remaining effort: 1-2 days**
