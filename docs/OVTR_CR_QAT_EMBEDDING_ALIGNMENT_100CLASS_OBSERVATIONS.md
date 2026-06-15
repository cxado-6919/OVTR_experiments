# OVTR CR-QAT/QAT/FP32 100-Class Embedding Alignment Observations

이 문서는 OVTR `lite` 모델에서 생성한 CR-QAT/QAT/FP32 embedding alignment 시각화 산출물을 기준으로 100개 클래스 관측을 정리한 것입니다.

## Source

- 산출물 디렉터리: `ovtr/results/embedding_viz_lite_qat_exp_a1_to_b_val/embedding_viz`
- 선택 클래스: `selected_classes.csv`
- 클래스별 지표: `metrics_summary_by_class.csv`
- 모델별 macro 평균: `metrics_model_macro_mean.csv`
- 실행 요약: `alignment_summary.json`
- 전체 scatter: `distortion_scatter_100classes.png`
- per-class 예시: `embedding_viz/<class_name>/examples/*.png`
- per-example class score 막대 그래프: `embedding_viz/<class_name>/examples/*_class_scores.png`
- per-example class score CSV: `embedding_viz/<class_name>/examples/*_class_scores.csv`
- class score 클래스별 요약: `class_score_summary_by_class.csv`
- class score 모델별 macro 요약: `class_score_model_macro.csv`

실행 요약 기준 설정은 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| requested classes | 100 |
| generated classes | 100 |
| available positive classes | 878 |
| comparison labels | `qat`, `cr_qat` |
| total groups | 1,983 |
| total positive regions | 23,106 |
| max groups per class | 20 |
| max examples per class | 2 |
| max regions per group | 32 |
| score temperature | 0.007 |
| visualization style | `ovtrack_alignment_distortion` |

## Metric Definitions

- `region_text_mae`: FP32와 비교한 target class raw alignment cosine의 MAE입니다. 낮을수록 FP32 region-text alignment를 더 잘 보존한 것입니다.
- `region_region_mae`: 같은 클래스 positive region embedding들 사이의 off-diagonal pairwise cosine matrix를 FP32와 비교한 MAE입니다. 낮을수록 FP32 region-region 관계를 더 잘 보존한 것입니다.
- `region_region_pearson`: FP32 pairwise relation 값과 비교 모델 pairwise relation 값의 Pearson correlation입니다. 높을수록 FP32 관계 구조와 방향성이 더 유사합니다.
- `mean_confidence`: temperature 0.007을 적용한 foreground class anchor softmax에서 target class 평균 confidence입니다. 높을수록 target class로 더 강하게 모이지만, FP32 보존 지표는 아닙니다.
- `mean_region_text`: target class raw alignment cosine 평균입니다. 높을수록 target class text embedding과 가깝지만, FP32 대비 왜곡량은 `region_text_mae`로 판단해야 합니다.
- `target_mean_cosine_score`: 저장된 positive query embedding과 target class anchor 사이의 평균 cosine score입니다. 높을수록 target class anchor와 가깝습니다.
- `target_margin_vs_best_selected_competitor`: target 평균 class score에서 선택된 foreground 경쟁 클래스 중 최고 평균 score를 뺀 값입니다. 높을수록 target class가 경쟁 클래스보다 앞섭니다.
- `target_is_top_mean_score`: 예제 단위 평균 class score에서 target이 선택된 경쟁 클래스 전체보다 높으면 1, 아니면 0입니다. 클래스별 값은 예제 평균입니다.
- `mean_target_rank`: region별 전체 foreground class score에서 target class의 평균 rank입니다. 낮을수록 좋습니다.
- `mean_target_rank_normalized`: `mean_target_rank / 1203`입니다. 낮을수록 좋습니다.

## Model-Level Summary

| model | classes | groups | regions | region_text_mae | region_region_mae | region_region_pearson | mean_confidence | mean_region_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 100 | 1,983 | 23,106 | 0.00000 | 0.00000 | 1.00000 | 0.07149 | 0.98112 |
| QAT | 100 | 1,983 | 23,106 | 0.02952 | 0.01633 | 0.00850 | 0.00632 | 0.95220 |
| CR-QAT | 100 | 1,983 | 23,106 | 0.02912 | 0.01959 | -0.00264 | 0.00929 | 0.95255 |

## Class-Score Analysis

OVTrack의 class score 막대 그래프 방식에 맞춰, 저장된 positive query embedding을 전체 OVTR class anchor와 cosine 비교했습니다. 각 예제에서는 target class, 모델별 top-k foreground 경쟁 클래스, 그리고 background가 있으면 background를 함께 그리도록 구현했습니다. 현재 OVTR 산출물에서는 class anchor shape이 `(1203, 512)`이고 LVIS foreground class 수도 1,203개라 extra background row가 없습니다. 따라서 이 100-class 분석의 class score 그래프와 CSV에는 background row가 없습니다.

산출물은 100개 클래스에서 클래스당 2개 예제, 총 200개 예제 기준입니다. 생성된 파일은 다음과 같습니다.

| artifact | count |
|---|---:|
| `*_class_scores.png` | 200 |
| `*_class_scores.csv` | 200 |
| background rows | 0 |

### Class-Score Model Macro Summary

아래 값은 예제별 target row를 클래스 단위로 평균한 뒤 100개 클래스 macro 평균을 낸 것입니다. `target_top_mean_score`는 예제 평균 class score 기준에서 target class가 선택된 경쟁 클래스보다 높은 비율입니다.

| model | classes | target cosine | margin vs best competitor | target top mean score | target softmax conf | target rank | rank normalized |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 100 | 0.98112 | -0.01085 | 70.5% | 0.07455 | 127.12 | 0.10567 |
| QAT | 100 | 0.95245 | -0.03343 | 7.0% | 0.00698 | 324.31 | 0.26958 |
| CR-QAT | 100 | 0.95409 | -0.02996 | 8.0% | 0.01072 | 311.02 | 0.25854 |

### CR-QAT vs QAT Class-Score Deltas

`delta = CR-QAT - QAT`입니다. rank 계열은 낮을수록 좋으므로 음수 delta가 개선입니다.

| metric | better direction | QAT | CR-QAT | delta | CR-QAT better classes | QAT better classes | ties |
|---|---|---:|---:|---:|---:|---:|---:|
| target cosine | higher | 0.95245 | 0.95409 | +0.00164 | 48 | 52 | 0 |
| margin vs best competitor | higher | -0.03343 | -0.02996 | +0.00347 | 51 | 49 | 0 |
| target top mean score | higher | 7.0% | 8.0% | +1.0pp | 6 | 6 | 88 |
| target softmax confidence | higher | 0.00698 | 0.01072 | +0.00375 | 60 | 40 | 0 |
| target rank | lower | 324.31 | 311.02 | -13.28 | 54 | 46 | 0 |
| rank normalized | lower | 0.26958 | 0.25854 | -0.01104 | 54 | 46 | 0 |

이 결과는 기존 embedding alignment 지표보다 CR-QAT의 작은 이득을 더 잘 보여줍니다. region-region 관계 보존은 QAT보다 나빠진 클래스가 많지만, class-score 관점에서는 CR-QAT이 target confidence를 높이고 target rank를 평균 13.28 class 정도 올립니다. 다만 target이 평균 score top인 비율은 QAT 7.0%에서 CR-QAT 8.0%로 1.0pp만 개선되어, class-score 개선도 강한 효과라기보다는 약한 ranking bias 개선에 가깝습니다.

성능 향상의 원인에 대한 해석은 추측입니다. TETA/ClsA는 전체 embedding geometry의 보존보다 최종 class score ordering과 target confidence 변화에 더 민감할 수 있습니다. 이 관점에서 보면 CR-QAT은 region-region 구조를 더 잘 보존하지는 못했지만, 일부 주요 클래스에서 target rank와 경쟁 클래스 대비 margin을 개선해 class assignment 오류를 줄였을 가능성이 있습니다.

### Largest CR-QAT Class-Score Gains

#### Target Rank

낮을수록 좋습니다. `delta = CR-QAT - QAT`입니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| dining_table | 921.40 | 456.85 | -464.55 |
| plate | 510.08 | 169.17 | -340.91 |
| truck | 506.83 | 322.25 | -184.58 |
| griddle | 888.97 | 714.50 | -174.48 |
| sunglasses | 721.92 | 547.80 | -174.13 |
| trousers | 210.95 | 64.60 | -146.35 |
| curtain | 271.42 | 127.08 | -144.33 |
| pug-dog | 1164.32 | 1036.34 | -127.98 |
| backpack | 200.16 | 86.13 | -114.03 |
| cushion | 414.50 | 301.88 | -112.62 |

#### Margin vs Best Selected Competitor

높을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| pug-dog | -0.31039 | -0.15198 | +0.15842 |
| dining_table | -0.10081 | -0.04528 | +0.05554 |
| houseboat | -0.27500 | -0.22877 | +0.04623 |
| stepladder | -0.09775 | -0.05354 | +0.04420 |
| truck | -0.05038 | -0.02593 | +0.02445 |
| sunglasses | -0.05430 | -0.03818 | +0.01613 |
| plate | -0.02236 | -0.00632 | +0.01604 |
| bottle | -0.01958 | -0.00592 | +0.01367 |
| car_(automobile) | -0.00833 | 0.00355 | +0.01188 |
| stylus | -0.05793 | -0.04664 | +0.01129 |

#### Target Cosine

높을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| pug-dog | 0.67551 | 0.80829 | +0.13278 |
| dining_table | 0.88932 | 0.94037 | +0.05105 |
| stepladder | 0.89290 | 0.92990 | +0.03700 |
| houseboat | 0.71696 | 0.75374 | +0.03678 |
| plate | 0.94366 | 0.97649 | +0.03283 |
| truck | 0.93089 | 0.95537 | +0.02447 |
| sunglasses | 0.92910 | 0.94887 | +0.01977 |
| griddle | 0.89925 | 0.91889 | +0.01965 |
| car_(automobile) | 0.97229 | 0.98482 | +0.01253 |
| cone | 0.97063 | 0.98244 | +0.01181 |

### Largest CR-QAT Class-Score Regressions

#### Target Rank

낮을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| Sharpie | 251.53 | 546.51 | +294.98 |
| crumb | 490.82 | 667.68 | +176.86 |
| boat | 258.58 | 430.84 | +172.26 |
| hinge | 281.38 | 423.12 | +141.75 |
| bolt | 102.70 | 230.36 | +127.66 |
| doorknob | 96.83 | 220.83 | +124.00 |
| bench | 65.98 | 189.67 | +123.69 |
| sheep | 383.27 | 501.91 | +118.64 |
| blinker | 134.96 | 249.56 | +114.60 |
| taillight | 180.53 | 282.82 | +102.28 |

#### Margin vs Best Selected Competitor

높을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| sheep | -0.03640 | -0.05647 | -0.02007 |
| boat | -0.01736 | -0.03370 | -0.01634 |
| crumb | -0.03674 | -0.05089 | -0.01415 |
| belt | -0.02168 | -0.03561 | -0.01393 |
| scarecrow | -0.11078 | -0.12456 | -0.01378 |
| Sharpie | -0.02267 | -0.03474 | -0.01207 |
| bench | -0.00773 | -0.01706 | -0.00933 |
| earring | -0.03830 | -0.04699 | -0.00869 |
| necktie | -0.00232 | -0.00936 | -0.00704 |
| vent | -0.02763 | -0.03382 | -0.00619 |

### Class-Score Interpretation

class-score 분석을 기존 alignment 분석과 함께 보면 다음처럼 정리할 수 있습니다.

- CR-QAT은 FP32 embedding geometry 전체를 QAT보다 안정적으로 보존하지는 않습니다. 특히 region-region MAE는 QAT보다 악화된 클래스가 많습니다.
- 그러나 target class score 자체, target confidence, target rank는 CR-QAT이 평균적으로 조금 더 좋습니다.
- 성능 개선 원인에 대한 추측으로는, OVTR 평가에서 실제 class assignment는 full embedding relation 보존보다 target class가 경쟁 class보다 조금이라도 위로 올라오는지에 더 직접적으로 영향을 받았을 수 있습니다.
- 따라서 현재 결과는 “CR-QAT이 embedding alignment 분석에서는 약해 보이지만, class-score/ranking 쪽으로는 작은 이득을 만들어 최종 ClsA/TETA 개선으로 이어졌을 가능성이 있다”로 해석하는 것이 가장 보수적입니다.

## Main Observations

1. CR-QAT의 region-text MAE는 QAT보다 아주 조금 낮습니다.
   - QAT: 0.02952
   - CR-QAT: 0.02912
   - macro 평균 차이: -0.00039
   - 클래스 승패: CR-QAT 우세 48개, QAT 우세 52개

2. CR-QAT의 region-region MAE는 QAT보다 높습니다.
   - QAT: 0.01633
   - CR-QAT: 0.01959
   - macro 평균 차이: +0.00326
   - 클래스 승패: CR-QAT 우세 27개, QAT 우세 73개

3. region-region Pearson은 CR-QAT이 일관되게 좋아졌다고 보기 어렵습니다.
   - QAT: 0.00850
   - CR-QAT: -0.00264
   - macro 평균 차이: -0.01114
   - 클래스 승패: CR-QAT 우세 50개, QAT 우세 50개

4. target class confidence는 CR-QAT이 더 자주 높습니다.
   - QAT: 0.00632
   - CR-QAT: 0.00929
   - macro 평균 차이: +0.00297
   - 클래스 승패: CR-QAT 우세 60개, QAT 우세 38개, 동률 2개

5. raw target cosine 평균은 두 quant 모델이 거의 같습니다.
   - QAT: 0.95220
   - CR-QAT: 0.95255
   - macro 평균 차이: +0.00035
   - 클래스 승패: CR-QAT 우세 49개, QAT 우세 51개

## Region-Text vs Region-Region Tradeoff

CR-QAT이 두 왜곡 지표를 동시에 개선한 클래스는 제한적입니다.

| category | class count |
|---|---:|
| CR-QAT improves both region-text MAE and region-region MAE | 17 |
| CR-QAT improves region-text only | 31 |
| CR-QAT improves region-region only | 10 |
| CR-QAT is worse on both | 42 |

따라서 이 OVTR positive-query 분석에서는 CR-QAT이 CR-QAT 논문식 기대처럼 region-text alignment와 region-region relation을 동시에 안정적으로 보존한다고 보기는 어렵습니다. 관측상 CR-QAT은 target confidence를 올리는 경향은 있으나, region-region 관계 보존은 QAT보다 악화되는 클래스가 더 많습니다.

## Largest CR-QAT Improvements

### Region-Text MAE

낮을수록 좋습니다. `delta = CR-QAT - QAT`입니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| pug-dog | 0.19555 | 0.13204 | -0.06351 |
| stepladder | 0.05018 | 0.02870 | -0.02148 |
| griddle | 0.04304 | 0.02450 | -0.01854 |
| plate | 0.04809 | 0.03094 | -0.01715 |
| sunglasses | 0.06752 | 0.05101 | -0.01651 |
| bead | 0.03955 | 0.02727 | -0.01228 |
| dining_table | 0.09279 | 0.08341 | -0.00938 |
| lightbulb | 0.03891 | 0.03006 | -0.00885 |
| police_cruiser | 0.07222 | 0.06354 | -0.00868 |
| houseboat | 0.08249 | 0.07445 | -0.00805 |

### Region-Region MAE

낮을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| belt | 0.03259 | 0.01488 | -0.01771 |
| chair | 0.02733 | 0.01523 | -0.01210 |
| sunglasses | 0.02165 | 0.01348 | -0.00817 |
| ball | 0.01949 | 0.01229 | -0.00720 |
| halter_top | 0.01688 | 0.01076 | -0.00612 |
| bead | 0.03066 | 0.02479 | -0.00588 |
| cup | 0.02709 | 0.02154 | -0.00555 |
| bird | 0.01264 | 0.00760 | -0.00504 |
| earring | 0.02723 | 0.02235 | -0.00488 |
| manhole | 0.02223 | 0.01758 | -0.00465 |

### Target Confidence

높을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| car_(automobile) | 0.04383 | 0.13076 | +0.08693 |
| plate | 0.00828 | 0.03607 | +0.02779 |
| jersey | 0.02026 | 0.04565 | +0.02539 |
| person | 0.04138 | 0.06381 | +0.02243 |
| truck | 0.00483 | 0.02277 | +0.01794 |
| curtain | 0.00455 | 0.01761 | +0.01305 |
| boat | 0.01549 | 0.02836 | +0.01286 |
| bicycle | 0.00011 | 0.01271 | +0.01259 |
| dining_table | 0.00009 | 0.01251 | +0.01242 |
| cushion | 0.00387 | 0.01313 | +0.00926 |

## Largest CR-QAT Regressions

### Region-Text MAE

낮을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| crumb | 0.03839 | 0.05595 | +0.01756 |
| newspaper | 0.01769 | 0.03470 | +0.01701 |
| sheep | 0.04674 | 0.06274 | +0.01600 |
| button | 0.04159 | 0.05621 | +0.01462 |
| earring | 0.05516 | 0.06687 | +0.01171 |
| Sharpie | 0.01050 | 0.02089 | +0.01039 |
| bolo_tie | 0.01408 | 0.02405 | +0.00997 |
| handle | 0.01390 | 0.02378 | +0.00988 |
| comic_book | 0.02291 | 0.03247 | +0.00956 |
| ring | 0.03545 | 0.04427 | +0.00883 |

### Region-Region MAE

낮을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| comic_book | 0.00522 | 0.02441 | +0.01918 |
| pug-dog | 0.01225 | 0.02955 | +0.01730 |
| painting | 0.00603 | 0.02305 | +0.01702 |
| bottle | 0.01514 | 0.02990 | +0.01476 |
| bamboo | 0.00835 | 0.02203 | +0.01368 |
| dining_table | 0.02345 | 0.03646 | +0.01300 |
| newspaper | 0.01027 | 0.02296 | +0.01268 |
| Sharpie | 0.00933 | 0.02104 | +0.01171 |
| coin | 0.00995 | 0.02136 | +0.01140 |
| bicycle | 0.00897 | 0.02027 | +0.01130 |

### Target Confidence

높을수록 좋습니다.

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| bench | 0.01060 | 0.00279 | -0.00781 |
| wheel | 0.01370 | 0.00888 | -0.00481 |
| sheep | 0.00462 | 0.00002 | -0.00460 |
| button | 0.00771 | 0.00326 | -0.00445 |
| traffic_light | 0.01465 | 0.01029 | -0.00436 |
| earring | 0.00432 | 0.00081 | -0.00351 |
| handle | 0.01137 | 0.00832 | -0.00306 |
| duct_tape | 0.00779 | 0.00501 | -0.00278 |
| ring | 0.00441 | 0.00189 | -0.00252 |
| ball | 0.00423 | 0.00188 | -0.00235 |

## Pearson Outliers

Pearson은 높을수록 FP32 relation pattern과 방향성이 비슷합니다.

### Largest CR-QAT Pearson Gains

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| stepladder | -0.20867 | 0.09853 | +0.30720 |
| bolo_tie | -0.19317 | 0.07542 | +0.26858 |
| awning | -0.01345 | 0.18635 | +0.19979 |
| bench | 0.03388 | 0.23151 | +0.19763 |
| police_cruiser | -0.07789 | 0.11414 | +0.19203 |
| cone | -0.03281 | 0.12379 | +0.15660 |
| handle | 0.00083 | 0.13867 | +0.13784 |
| pug-dog | 0.00776 | 0.13210 | +0.12434 |
| glove | -0.14982 | -0.03506 | +0.11476 |
| earring | -0.05743 | 0.05286 | +0.11029 |

### Largest CR-QAT Pearson Losses

| class | QAT | CR-QAT | delta |
|---|---:|---:|---:|
| dining_table | 0.18617 | -0.21638 | -0.40255 |
| beret | 0.03632 | -0.30083 | -0.33715 |
| manhole | 0.20446 | -0.10598 | -0.31044 |
| tag | 0.14672 | -0.15469 | -0.30141 |
| doorknob | 0.05841 | -0.16928 | -0.22769 |
| brake_light | 0.19730 | 0.01441 | -0.18288 |
| paperback_book | 0.23504 | 0.05360 | -0.18143 |
| lamppost | 0.10674 | -0.07040 | -0.17714 |
| boat | 0.13428 | -0.03832 | -0.17260 |
| nut | 0.05690 | -0.10862 | -0.16553 |

## Notes on FP32 Reference Matrices

Per-class example 이미지의 `reference` region-region heatmap이 대부분 빨갛게 보이는 것은 현재 산출물이 raw same-class positive query embedding 간 cosine matrix를 직접 그리기 때문입니다. 선택된 FP32 positive query들은 같은 target class에 대해 이미 높은 cosine 유사도를 가지므로, absolute matrix가 0.97-1.00 범위에 몰리면 거의 전부 붉게 보일 수 있습니다.

이 현상만으로 오류라고 판단하기는 어렵습니다. 관계 구조 왜곡은 absolute heatmap 색상보다 FP32 대비 `region_region_mae`, `region_region_pearson`, 또는 차분 matrix로 보는 것이 더 적합합니다.

## Interpretation

이 100-class OVTR positive-query 분석에서는 CR-QAT의 효과가 CR-QAT 논문의 OV-detection 설정과 다르게 나타납니다.

- CR-QAT은 target class confidence를 QAT보다 높이는 경향이 있습니다.
- class-score 분석에서도 CR-QAT은 target rank와 경쟁 class 대비 margin을 평균적으로 조금 개선합니다.
- 하지만 FP32 대비 region-region relation 보존은 QAT보다 나빠진 클래스가 더 많습니다.
- region-text 보존은 평균상 CR-QAT이 아주 근소하게 좋지만, 클래스별 승패는 거의 반반입니다.
- 두 왜곡 지표를 동시에 개선한 클래스는 17/100개뿐이며, 둘 다 악화된 클래스는 42/100개입니다.
- 최종 성능 개선 원인은 추측이지만, OVTR의 class assignment가 relation 보존보다 target class ranking 개선에 더 민감했기 때문일 가능성이 있습니다.

따라서 현재 결과는 "OVTR에 적용한 CR-QAT이 text alignment confidence에는 일부 이득을 줄 수 있지만, OV-MOT query embedding의 region-region relational structure 보존까지 일관되게 개선하지는 못한다"는 관측으로 정리하는 것이 타당합니다.

## Limitations

- 이 분석은 FP32 post-process OVTR positive query를 기준으로 QAT/CR-QAT을 비교합니다. CR-QAT 논문이 목표로 한 OV-detection RoI/region embedding 분석과 완전히 같은 조건은 아닙니다.
- 기본 sequence sampling으로 수집한 70개 sequence, 2,501 frame 기반 결과입니다. 전체 validation set 전체에 대한 결론으로 확장하려면 sampling 없이 재실행한 결과가 필요합니다.
- `mean_confidence`와 class-score softmax confidence는 temperature softmax confidence이므로 alignment 보존 지표가 아니라 class selection sharpness에 가까운 보조 지표입니다.
- class-score margin은 저장된 그래프의 선택 class set 안에서 계산했습니다. 각 모델의 top-k 경쟁 class가 포함되므로 best selected competitor는 예제별 주요 경쟁 class를 보지만, 전체 1,203개 class의 모든 margin 분포를 저장한 것은 아닙니다.

## Follow-Up Checks

- per-class example에 `model - FP32` 차분 heatmap을 추가하면 reference saturation 문제를 더 직접적으로 확인할 수 있습니다.
- class-score 분석을 더 엄밀히 하려면 예제별 전체 1,203개 class score vector를 저장해 full top-k/rank/margin 분포를 직접 비교하는 것이 좋습니다.
- OV-MOT 특화 관측을 위해 detection query embedding뿐 아니라 track embedding, association feature, temporal consistency metric을 별도로 비교하는 것이 좋습니다.
- CR-QAT curriculum loss가 실제로 어느 module/embedding에 강하게 작용했는지 확인하려면 checkpoint별 quant manifest와 training loss log를 함께 대조해야 합니다.
