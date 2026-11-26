# Preregistration of the listening test

This is a copy of the text submitted to OSF Registries on 20 November 2026, before any listening data were collected. The authoritative, time-stamped version is the OSF registration (osf.io/mky35; under embargo during review, view-only link: https://osf.io/mky35/overview?view_only=ac181516e6ca492bb75983ae44d59f96).

---

**Title.** Perceptual evaluation of singing-technique control in a frozen zero-shot singing voice converter.

**Timing.** Registered before any listening data were collected. Objective evaluations of the same system (distances to real technique recordings, a technique classifier, conversion between singers and sung lyric phrases) had been completed at the time of registration; they involve no listeners.

**Background.** A technique-gated low-rank adapter adds continuous control of five singing techniques (breathy, belt, vocal fry, forte, piano) to a frozen Seed-VC converter. The study asks which of these controls listeners identify as the intended technique, and whether listeners judge them as more like a change of technique than a matched static equaliser derived from the same output.

**Hypotheses.** Formed after informal pilot listening by the author; the present study is an independent test.
- H1. Breathy and vocal-fry control outputs are identified as the intended technique, and are judged more like a change of technique than their matched equaliser.
- H2. Belt, forte and piano control outputs are not reliably distinguished from their matched equaliser in that judgement.
- H3. The magnitude of change in cepstral peak prominence or in a high-frequency energy ratio does not predict, across stimuli, whether listeners identify the intended technique.

**Design.** Web-based, self-paced, offline test; all audio set to one common RMS level; trial order and A/B order randomised per listener.
- Part I, identification (39 trials): unmodified output vs. one processed sample. Conditions: technique adapter (20 trials), real recordings of held-out singers, plain vs. technique take (10), technique present in the source recording (5), training-free contrastive guidance (4). Questions: noticeability (3 levels), kind of change (9 options, one per technique plus no change, post-processing, other technique, different singer), audio quality (3 levels), same singer (3 levels), willingness to use (3 levels).
- Part II, matched equaliser (14 trials): unmodified output, then the processed sample and its matched equaliser in random order (10 adapter, 2 guidance, 2 calibration trials with identical A and B). Questions: difference between A and B, which is more like the singer changing technique rather than post-processing (A, B, both, neither, cannot tell), which they would use, quality of A and of B.

**Participants.** Adult volunteers from two groups: students in computer science and related fields, and creators of virtual-singer, singing-synthesis or singing-voice-conversion content. Target: at least 20 included listeners, about half from each group. The author does not take part. Informed consent is obtained on the first page; no identifying information is collected.

**Exclusion.** Listeners who rate both calibration pairs (identical A and B) as clearly different are excluded entirely; the number excluded is reported. Incomplete submissions are analysed for the trials answered.

**Analysis.**
- Identification: per condition and technique, the proportion of responses naming the intended change and the full response distribution. Test of the intended response against the most frequent alternative (taken from the pooled responses): per-listener difference of the two proportions, one-sided Wilcoxon signed-rank test across listeners; secondary estimate from a Bayesian logistic regression with crossed random intercepts for listeners and stimuli, fitted on the responses that are either of the two options.
- Matched equaliser: proportion of decided responses (processed sample or equaliser) naming the processed sample as more like a change of technique; the same two tests against the equaliser.
- Holm correction over the five techniques within each condition; α = 0.05.
- H1 is supported for a technique if both the identification test and the matched-equaliser test are significant after correction. H2 is supported for a technique if the matched-equaliser test is not significant; this does not establish equivalence, and the observed proportion with its interval is reported.
- H3: Spearman correlation across identification stimuli between the absolute change in each descriptor and the identification rate.
- Results are reported separately for the two listener groups as a descriptive analysis.

**Related objective predictions.** Recorded on 2026-11-19 before the first objective experiment was run (comparison with real technique recordings of held-out singers): O1, for breathy and vocal fry the correct control lowers the equaliser-invariant part of the cepstral distance to the real technique recording relative to the unmodified output, more than the four other controls, and a classifier without static spectral features assigns the output to the intended technique more often than the unmodified output; O2, for belt, forte and piano most of the reduction in distance lies in the static part.
