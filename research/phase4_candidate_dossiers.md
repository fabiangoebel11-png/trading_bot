# Phase 4.2 Candidate Dossiers

These 31 entries are diagnostic only. They do not rank or promote models.

## Candidate 1

- asset: BTC/USDT
- timeframe: 1h
- horizon: 12h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5449992541350507,
    "metric_pr_auc": 0.5087493549562444,
    "metric_balanced_accuracy": 0.5326588125945118,
    "metric_log_loss": 0.6911623944358447,
    "metric_brier_score": 0.2489530117822077,
    "metric_accuracy": 0.5305936073059361,
    "metric_mean_forward_return": -0.0011562891748861,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4692922374429223,
    "metric_signal_coverage_up_0_5": 0.5316210045662101,
    "metric_signal_coverage_down_0_5": 0.4683789954337899,
    "metric_signal_coverage_up_0_55": 0.1626712328767123,
    "metric_signal_coverage_down_0_55": 0.1238584474885844,
    "metric_signal_coverage_up_0_6": 0.0457762557077625,
    "metric_signal_coverage_down_0_6": 0.0157534246575342,
    "metric_signal_coverage_up_0_65": 0.0147260273972602,
    "metric_signal_coverage_down_0_65": 0.0019406392694063,
    "metric_signal_coverage_up_0_7": 0.0045662100456621,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0004431454993327,
    "metric_mean_forward_return_down_0_5": 0.0019657237586182,
    "metric_mean_forward_return_up_0_55": 0.0003804091025871,
    "metric_mean_forward_return_down_0_55": 0.0033945457231924,
    "metric_mean_forward_return_up_0_6": 0.0016025638698683,
    "metric_mean_forward_return_down_0_6": 0.0106366502410411,
    "metric_mean_forward_return_up_0_65": -0.0034935856424434,
    "metric_mean_forward_return_down_0_65": 0.0202626391646184,
    "metric_mean_forward_return_up_0_7": -0.0177791796044209,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5566811247210803,
    "metric_pr_auc": 0.5655297123291217,
    "metric_balanced_accuracy": 0.5408882671531032,
    "metric_log_loss": 0.6886065930737907,
    "metric_brier_score": 0.2477410024838234,
    "metric_accuracy": 0.5395593104235643,
    "metric_mean_forward_return": 0.0014174688604865,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5183240095901359,
    "metric_signal_coverage_up_0_5": 0.4652357575065646,
    "metric_signal_coverage_down_0_5": 0.5347642424934353,
    "metric_signal_coverage_up_0_55": 0.0711268409635803,
    "metric_signal_coverage_down_0_55": 0.0768352551661148,
    "metric_signal_coverage_up_0_6": 0.0130151843817787,
    "metric_signal_coverage_down_0_6": 0.0060509190546866,
    "metric_signal_coverage_up_0_65": 0.0017125242607603,
    "metric_signal_coverage_down_0_65": 0.0004566731362027,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0021358268485855,
    "metric_mean_forward_return_down_0_5": -0.0007925096799777,
    "metric_mean_forward_return_up_0_55": 0.0049140832013455,
    "metric_mean_forward_return_down_0_55": -0.0007489944446131,
    "metric_mean_forward_return_up_0_6": 0.0083398821873927,
    "metric_mean_forward_return_down_0_6": 0.0072375903201462,
    "metric_mean_forward_return_up_0_65": 0.0148124088343639,
    "metric_mean_forward_return_down_0_65": 0.0089357628212117,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5332824503076958,
    "metric_pr_auc": 0.5667934996633534,
    "metric_balanced_accuracy": 0.5338145977565616,
    "metric_log_loss": 0.6927725805454802,
    "metric_brier_score": 0.2498019073027399,
    "metric_accuracy": 0.530851548269581,
    "metric_mean_forward_return": 0.001261483688387,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5329007285974499,
    "metric_signal_coverage_up_0_5": 0.4571948998178506,
    "metric_signal_coverage_down_0_5": 0.5428051001821493,
    "metric_signal_coverage_up_0_55": 0.1096311475409836,
    "metric_signal_coverage_down_0_55": 0.1300091074681238,
    "metric_signal_coverage_up_0_6": 0.0214025500910746,
    "metric_signal_coverage_down_0_6": 0.0152550091074681,
    "metric_signal_coverage_up_0_65": 0.002959927140255,
    "metric_signal_coverage_down_0_65": 0.0012522768670309,
    "metric_signal_coverage_up_0_7": 0.0003415300546448,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0021129675566128,
    "metric_mean_forward_return_down_0_5": -0.0005442942557539,
    "metric_mean_forward_return_up_0_55": 0.0031232771373726,
    "metric_mean_forward_return_down_0_55": -0.0007383570667062,
    "metric_mean_forward_return_up_0_6": 0.0029398060599175,
    "metric_mean_forward_return_down_0_6": 0.0011483420167676,
    "metric_mean_forward_return_up_0_65": 0.0090252292977981,
    "metric_mean_forward_return_down_0_65": -0.0104088587131676,
    "metric_mean_forward_return_up_0_7": 0.0336621200318835,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5106290493434829,
    "metric_pr_auc": 0.5228997547288545,
    "metric_balanced_accuracy": 0.5058457299112091,
    "metric_log_loss": 0.6949454760100994,
    "metric_brier_score": 0.2508917341812337,
    "metric_accuracy": 0.504773902665429,
    "metric_mean_forward_return": 6.007763805394336e-05,
    "metric_sample_count": 15082.0,
    "metric_positive_rate": 0.5126641029041241,
    "metric_signal_coverage_up_0_5": 0.4578305264553772,
    "metric_signal_coverage_down_0_5": 0.5421694735446227,
    "metric_signal_coverage_up_0_55": 0.0997215223445166,
    "metric_signal_coverage_down_0_55": 0.1151704018034743,
    "metric_signal_coverage_up_0_6": 0.0116032356451399,
    "metric_signal_coverage_down_0_6": 0.0064315077575918,
    "metric_signal_coverage_up_0_65": 0.0006630420368651,
    "metric_signal_coverage_down_0_65": 0.0007956504442381,
    "metric_signal_coverage_up_0_7": 0.000132608407373,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -5.2475557203141904e-05,
    "metric_mean_forward_return_down_0_5": -0.000155122252613,
    "metric_mean_forward_return_up_0_55": 0.0003599386919466,
    "metric_mean_forward_return_down_0_55": 0.0007683181353496,
    "metric_mean_forward_return_up_0_6": 0.0022812739249262,
    "metric_mean_forward_return_down_0_6": 0.0061524892560047,
    "metric_mean_forward_return_up_0_65": 0.0354625802947418,
    "metric_mean_forward_return_down_0_65": 0.0168677327485864,
    "metric_mean_forward_return_up_0_7": 0.0473464298566844,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 2

- asset: BTC/USDT
- timeframe: 1h
- horizon: 1h
- model: hist_gradient_boosting
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2022; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5726248488577619,
    "metric_pr_auc": 0.5558840277528173,
    "metric_balanced_accuracy": 0.5537125475320055,
    "metric_log_loss": 0.686025291304026,
    "metric_brier_score": 0.2464513414219979,
    "metric_accuracy": 0.553082191780822,
    "metric_mean_forward_return": -9.407711287377576e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4974885844748858,
    "metric_signal_coverage_up_0_5": 0.6252283105022831,
    "metric_signal_coverage_down_0_5": 0.3747716894977169,
    "metric_signal_coverage_up_0_55": 0.3122146118721461,
    "metric_signal_coverage_down_0_55": 0.1519406392694064,
    "metric_signal_coverage_up_0_6": 0.0871004566210045,
    "metric_signal_coverage_down_0_6": 0.0299086757990867,
    "metric_signal_coverage_up_0_65": 0.0091324200913242,
    "metric_signal_coverage_down_0_65": 0.0018264840182648,
    "metric_signal_coverage_up_0_7": 0.0001141552511415,
    "metric_signal_coverage_down_0_7": 0.0001141552511415,
    "metric_mean_forward_return_up_0_5": -2.2744586950359675e-05,
    "metric_mean_forward_return_down_0_5": 0.0002130805379369,
    "metric_mean_forward_return_up_0_55": -0.0001158143013959,
    "metric_mean_forward_return_down_0_55": 0.0001532743982582,
    "metric_mean_forward_return_up_0_6": 7.029475409341704e-05,
    "metric_mean_forward_return_down_0_6": 0.000424888801741,
    "metric_mean_forward_return_up_0_65": 0.0013966227616996,
    "metric_mean_forward_return_down_0_65": 0.0009084899131318,
    "metric_mean_forward_return_up_0_7": 0.0332649244785423,
    "metric_mean_forward_return_down_0_7": -0.0039886414775551
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5700130709173965,
    "metric_pr_auc": 0.5791350462830955,
    "metric_balanced_accuracy": 0.5470440327964615,
    "metric_log_loss": 0.6852590045035085,
    "metric_brier_score": 0.2460851233803602,
    "metric_accuracy": 0.5483502682954675,
    "metric_mean_forward_return": 0.0001174623817694,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5134147733759562,
    "metric_signal_coverage_up_0_5": 0.5499486242721772,
    "metric_signal_coverage_down_0_5": 0.4500513757278228,
    "metric_signal_coverage_up_0_55": 0.2624728850325379,
    "metric_signal_coverage_down_0_55": 0.147277086425391,
    "metric_signal_coverage_up_0_6": 0.0496632035620504,
    "metric_signal_coverage_down_0_6": 0.0077634433154469,
    "metric_signal_coverage_up_0_65": 0.0011416828405069,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 8.018410696525084e-05,
    "metric_mean_forward_return_down_0_5": -0.0001630152609505,
    "metric_mean_forward_return_up_0_55": 0.0002312221334351,
    "metric_mean_forward_return_down_0_55": -0.0004476932350021,
    "metric_mean_forward_return_up_0_6": 0.0008367902342246,
    "metric_mean_forward_return_down_0_6": 7.473942080183143e-05,
    "metric_mean_forward_return_up_0_65": 0.0004755869128308,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5579369093423401,
    "metric_pr_auc": 0.5591005970227985,
    "metric_balanced_accuracy": 0.5472946004772021,
    "metric_log_loss": 0.6880570852104911,
    "metric_brier_score": 0.2474614121230174,
    "metric_accuracy": 0.5480418943533698,
    "metric_mean_forward_return": 0.0001062213431995,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5114981785063752,
    "metric_signal_coverage_up_0_5": 0.5335837887067395,
    "metric_signal_coverage_down_0_5": 0.4664162112932605,
    "metric_signal_coverage_up_0_55": 0.2882513661202185,
    "metric_signal_coverage_down_0_55": 0.2058287795992714,
    "metric_signal_coverage_up_0_6": 0.0694444444444444,
    "metric_signal_coverage_down_0_6": 0.0129781420765027,
    "metric_signal_coverage_up_0_65": 0.0009107468123861,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0002599038399675,
    "metric_mean_forward_return_down_0_5": 6.959263345441091e-05,
    "metric_mean_forward_return_up_0_55": 0.0003429886836021,
    "metric_mean_forward_return_down_0_55": 0.0001723144506954,
    "metric_mean_forward_return_up_0_6": 0.000605178171402,
    "metric_mean_forward_return_down_0_6": 0.0007506132054198,
    "metric_mean_forward_return_up_0_65": 0.0028832512501034,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.53699917568368,
    "metric_pr_auc": 0.5306469013803672,
    "metric_balanced_accuracy": 0.5281517614388113,
    "metric_log_loss": 0.6933655502713061,
    "metric_brier_score": 0.2500744616038411,
    "metric_accuracy": 0.528324388789505,
    "metric_mean_forward_return": 6.073548038943529e-06,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5016895249453389,
    "metric_signal_coverage_up_0_5": 0.5511826674617373,
    "metric_signal_coverage_down_0_5": 0.4488173325382628,
    "metric_signal_coverage_up_0_55": 0.2857616113430067,
    "metric_signal_coverage_down_0_55": 0.2062545550917644,
    "metric_signal_coverage_up_0_6": 0.0939508381368846,
    "metric_signal_coverage_down_0_6": 0.0240508845160007,
    "metric_signal_coverage_up_0_65": 0.0054992380573775,
    "metric_signal_coverage_down_0_65": 0.0002650235208374,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 5.6782140507843925e-05,
    "metric_mean_forward_return_down_0_5": 5.620055599837304e-05,
    "metric_mean_forward_return_up_0_55": 4.1842272195068826e-06,
    "metric_mean_forward_return_down_0_55": 9.607142735545076e-05,
    "metric_mean_forward_return_up_0_6": -0.0001004551120121,
    "metric_mean_forward_return_down_0_6": -0.0001130033714217,
    "metric_mean_forward_return_up_0_65": 0.0003875323104717,
    "metric_mean_forward_return_down_0_65": 0.0001538514950718,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 3

- asset: BTC/USDT
- timeframe: 1h
- horizon: 1h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5577090725376403,
    "metric_pr_auc": 0.5426134973530268,
    "metric_balanced_accuracy": 0.5406046398451703,
    "metric_log_loss": 0.6887302656588077,
    "metric_brier_score": 0.2477885485403801,
    "metric_accuracy": 0.5404109589041096,
    "metric_mean_forward_return": -9.407711287377576e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4974885844748858,
    "metric_signal_coverage_up_0_5": 0.5383561643835616,
    "metric_signal_coverage_down_0_5": 0.4616438356164383,
    "metric_signal_coverage_up_0_55": 0.1737442922374429,
    "metric_signal_coverage_down_0_55": 0.1264840182648401,
    "metric_signal_coverage_up_0_6": 0.0296803652968036,
    "metric_signal_coverage_down_0_6": 0.0154109589041095,
    "metric_signal_coverage_up_0_65": 0.0031963470319634,
    "metric_signal_coverage_down_0_65": 0.0011415525114155,
    "metric_signal_coverage_up_0_7": 0.0001141552511415,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -6.40948518909907e-05,
    "metric_mean_forward_return_down_0_5": 0.0001290415893314,
    "metric_mean_forward_return_up_0_55": -0.0001036465458095,
    "metric_mean_forward_return_down_0_55": -5.759630051624277e-07,
    "metric_mean_forward_return_up_0_6": -0.0013375205471096,
    "metric_mean_forward_return_down_0_6": -0.0001678818346563,
    "metric_mean_forward_return_up_0_65": 0.0003313050964989,
    "metric_mean_forward_return_down_0_65": 0.0026920485902822,
    "metric_mean_forward_return_up_0_7": 0.0648723290361514,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5628070833394639,
    "metric_pr_auc": 0.5679387232093743,
    "metric_balanced_accuracy": 0.5491752309558893,
    "metric_log_loss": 0.6873035315179603,
    "metric_brier_score": 0.2470950555326906,
    "metric_accuracy": 0.5497202877040758,
    "metric_mean_forward_return": 0.0001174623817694,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5134147733759562,
    "metric_signal_coverage_up_0_5": 0.5216348898276059,
    "metric_signal_coverage_down_0_5": 0.4783651101723941,
    "metric_signal_coverage_up_0_55": 0.1343760703276629,
    "metric_signal_coverage_down_0_55": 0.1141682840506907,
    "metric_signal_coverage_up_0_6": 0.0141568672222856,
    "metric_signal_coverage_down_0_6": 0.0057084142025345,
    "metric_signal_coverage_up_0_65": 0.0012558511245575,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0002283365681013,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 9.27931728182622e-05,
    "metric_mean_forward_return_down_0_5": -0.000144363006041,
    "metric_mean_forward_return_up_0_55": 0.000104409267239,
    "metric_mean_forward_return_down_0_55": -0.0004134327681387,
    "metric_mean_forward_return_up_0_6": 0.0006549928923012,
    "metric_mean_forward_return_down_0_6": 0.0002524029302368,
    "metric_mean_forward_return_up_0_65": 0.0060084051770206,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0215622842896689,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5444418757929098,
    "metric_pr_auc": 0.5487950983953214,
    "metric_balanced_accuracy": 0.5327323950879752,
    "metric_log_loss": 0.6903619093355734,
    "metric_brier_score": 0.2486039629106559,
    "metric_accuracy": 0.5322176684881603,
    "metric_mean_forward_return": 0.0001062213431995,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5114981785063752,
    "metric_signal_coverage_up_0_5": 0.4783697632058288,
    "metric_signal_coverage_down_0_5": 0.5216302367941712,
    "metric_signal_coverage_up_0_55": 0.1604052823315118,
    "metric_signal_coverage_down_0_55": 0.1590391621129326,
    "metric_signal_coverage_up_0_6": 0.0292577413479052,
    "metric_signal_coverage_down_0_6": 0.0138888888888888,
    "metric_signal_coverage_up_0_65": 0.0011384335154826,
    "metric_signal_coverage_down_0_65": 0.0006830601092896,
    "metric_signal_coverage_up_0_7": 0.0001138433515482,
    "metric_signal_coverage_down_0_7": 0.0001138433515482,
    "metric_mean_forward_return_up_0_5": 0.0001892850117371,
    "metric_mean_forward_return_down_0_5": -3.0046411904309147e-05,
    "metric_mean_forward_return_up_0_55": 0.0003843969480761,
    "metric_mean_forward_return_down_0_55": -6.476091796449426e-05,
    "metric_mean_forward_return_up_0_6": 0.0002555832084822,
    "metric_mean_forward_return_down_0_6": -8.734093857467286e-05,
    "metric_mean_forward_return_up_0_65": 0.0028621952024083,
    "metric_mean_forward_return_down_0_65": -0.0049116762539723,
    "metric_mean_forward_return_up_0_7": 0.0179887026049234,
    "metric_mean_forward_return_down_0_7": -0.0047984080405936
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5317918070290666,
    "metric_pr_auc": 0.5259479301565853,
    "metric_balanced_accuracy": 0.5235926305446703,
    "metric_log_loss": 0.692579513776464,
    "metric_brier_score": 0.2497074808459076,
    "metric_accuracy": 0.5235539654144306,
    "metric_mean_forward_return": 6.073548038943529e-06,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5016895249453389,
    "metric_signal_coverage_up_0_5": 0.4886371165440933,
    "metric_signal_coverage_down_0_5": 0.5113628834559067,
    "metric_signal_coverage_up_0_55": 0.1635195123567216,
    "metric_signal_coverage_down_0_55": 0.1647783740806996,
    "metric_signal_coverage_up_0_6": 0.0166964818127608,
    "metric_signal_coverage_down_0_6": 0.0089445438282647,
    "metric_signal_coverage_up_0_65": 0.0005300470416749,
    "metric_signal_coverage_down_0_65": 0.0004637911614655,
    "metric_signal_coverage_up_0_7": 6.625588020936858e-05,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 9.500710388210905e-06,
    "metric_mean_forward_return_down_0_5": -2.798694148577256e-06,
    "metric_mean_forward_return_up_0_55": 2.3589392426120168e-05,
    "metric_mean_forward_return_down_0_55": 8.368922413243549e-05,
    "metric_mean_forward_return_up_0_6": -0.0001582804663226,
    "metric_mean_forward_return_down_0_6": -0.000529372327865,
    "metric_mean_forward_return_up_0_65": 0.010222386888315,
    "metric_mean_forward_return_down_0_65": -6.53601208659506e-05,
    "metric_mean_forward_return_up_0_7": 0.0202663673823562,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 4

- asset: BTC/USDT
- timeframe: 1h
- horizon: 1h
- model: random_forest
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2022; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5701903615507908,
    "metric_pr_auc": 0.5539952136419578,
    "metric_balanced_accuracy": 0.5523096014390388,
    "metric_log_loss": 0.6865384452200911,
    "metric_brier_score": 0.2466720439937552,
    "metric_accuracy": 0.5521689497716895,
    "metric_mean_forward_return": -9.407711287377576e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4974885844748858,
    "metric_signal_coverage_up_0_5": 0.5277397260273973,
    "metric_signal_coverage_down_0_5": 0.4722602739726027,
    "metric_signal_coverage_up_0_55": 0.2845890410958904,
    "metric_signal_coverage_down_0_55": 0.2167808219178082,
    "metric_signal_coverage_up_0_6": 0.0757990867579908,
    "metric_signal_coverage_down_0_6": 0.0575342465753424,
    "metric_signal_coverage_up_0_65": 0.0126712328767123,
    "metric_signal_coverage_down_0_65": 0.0062785388127853,
    "metric_signal_coverage_up_0_7": 0.0039954337899543,
    "metric_signal_coverage_down_0_7": 0.0006849315068493,
    "metric_mean_forward_return_up_0_5": -1.841593448566944e-05,
    "metric_mean_forward_return_down_0_5": 0.0001786266965547,
    "metric_mean_forward_return_up_0_55": -6.486634050858925e-05,
    "metric_mean_forward_return_down_0_55": 0.0001717987702451,
    "metric_mean_forward_return_up_0_6": 0.0001986984585424,
    "metric_mean_forward_return_down_0_6": 0.0002364647044016,
    "metric_mean_forward_return_up_0_65": 0.0007045710342702,
    "metric_mean_forward_return_down_0_65": -0.000465957329148,
    "metric_mean_forward_return_up_0_7": 0.0004618329808503,
    "metric_mean_forward_return_down_0_7": -0.0037949828254446
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5655328172794063,
    "metric_pr_auc": 0.569513612482163,
    "metric_balanced_accuracy": 0.5469568220411188,
    "metric_log_loss": 0.6867286974364761,
    "metric_brier_score": 0.2467976363190484,
    "metric_accuracy": 0.5473227537390113,
    "metric_mean_forward_return": 0.0001174623817694,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5134147733759562,
    "metric_signal_coverage_up_0_5": 0.5148989610686151,
    "metric_signal_coverage_down_0_5": 0.4851010389313848,
    "metric_signal_coverage_up_0_55": 0.2569928073981048,
    "metric_signal_coverage_down_0_55": 0.218518095673022,
    "metric_signal_coverage_up_0_6": 0.0558282909007877,
    "metric_signal_coverage_down_0_6": 0.0482931841534421,
    "metric_signal_coverage_up_0_65": 0.0057084142025345,
    "metric_signal_coverage_down_0_65": 0.0038817216577234,
    "metric_signal_coverage_up_0_7": 0.0011416828405069,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 6.455139646642019e-05,
    "metric_mean_forward_return_down_0_5": -0.0001736234887868,
    "metric_mean_forward_return_up_0_55": 0.0001605102518291,
    "metric_mean_forward_return_down_0_55": -0.0002615970199127,
    "metric_mean_forward_return_up_0_6": 0.0002357725072688,
    "metric_mean_forward_return_down_0_6": -0.0003753886425242,
    "metric_mean_forward_return_up_0_65": 0.0014284240557515,
    "metric_mean_forward_return_down_0_65": 9.114536083867756e-05,
    "metric_mean_forward_return_up_0_7": -0.0022829348195218,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5572183727316472,
    "metric_pr_auc": 0.561811387835025,
    "metric_balanced_accuracy": 0.5450928794022945,
    "metric_log_loss": 0.6890390536573269,
    "metric_brier_score": 0.2479451909281368,
    "metric_accuracy": 0.5446265938069217,
    "metric_mean_forward_return": 0.0001062213431995,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5114981785063752,
    "metric_signal_coverage_up_0_5": 0.4807604735883424,
    "metric_signal_coverage_down_0_5": 0.5192395264116576,
    "metric_signal_coverage_up_0_55": 0.2743624772313296,
    "metric_signal_coverage_down_0_55": 0.285063752276867,
    "metric_signal_coverage_up_0_6": 0.0696721311475409,
    "metric_signal_coverage_down_0_6": 0.0725182149362477,
    "metric_signal_coverage_up_0_65": 0.0107012750455373,
    "metric_signal_coverage_down_0_65": 0.0048952641165755,
    "metric_signal_coverage_up_0_7": 0.000455373406193,
    "metric_signal_coverage_down_0_7": 0.0001138433515482,
    "metric_mean_forward_return_up_0_5": 0.0002140600586998,
    "metric_mean_forward_return_down_0_5": -6.3741834631669525e-06,
    "metric_mean_forward_return_up_0_55": 0.000366926815509,
    "metric_mean_forward_return_down_0_55": 8.535289577610144e-05,
    "metric_mean_forward_return_up_0_6": 0.0008487915832344,
    "metric_mean_forward_return_down_0_6": 0.0003207151607916,
    "metric_mean_forward_return_up_0_65": 0.0007656043144433,
    "metric_mean_forward_return_down_0_65": -5.74883970144403e-05,
    "metric_mean_forward_return_up_0_7": 0.0031801046139857,
    "metric_mean_forward_return_down_0_7": 0.0168783219768119
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.538647448352572,
    "metric_pr_auc": 0.5344545311067241,
    "metric_balanced_accuracy": 0.5301797860865436,
    "metric_log_loss": 0.6935870871732043,
    "metric_brier_score": 0.2501753989675218,
    "metric_accuracy": 0.5301795534353674,
    "metric_mean_forward_return": 6.073548038943529e-06,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5016895249453389,
    "metric_signal_coverage_up_0_5": 0.5000331279401047,
    "metric_signal_coverage_down_0_5": 0.4999668720598953,
    "metric_signal_coverage_up_0_55": 0.2764195322334857,
    "metric_signal_coverage_down_0_55": 0.2693301530510832,
    "metric_signal_coverage_up_0_6": 0.0777844033657987,
    "metric_signal_coverage_down_0_6": 0.0644007155635062,
    "metric_signal_coverage_up_0_65": 0.015106340687736,
    "metric_signal_coverage_down_0_65": 0.0047041674948651,
    "metric_signal_coverage_up_0_7": 0.0008613264427217,
    "metric_signal_coverage_down_0_7": 6.625588020936858e-05,
    "metric_mean_forward_return_up_0_5": 6.874610522907572e-05,
    "metric_mean_forward_return_down_0_5": 5.660731455235354e-05,
    "metric_mean_forward_return_up_0_55": 1.2410075026213788e-05,
    "metric_mean_forward_return_down_0_55": 7.008724134776528e-05,
    "metric_mean_forward_return_up_0_6": 8.266619009354958e-05,
    "metric_mean_forward_return_down_0_6": 7.910018801031466e-05,
    "metric_mean_forward_return_up_0_65": 0.0002240405823014,
    "metric_mean_forward_return_down_0_65": -0.0010106078185247,
    "metric_mean_forward_return_up_0_7": 0.002903450928583,
    "metric_mean_forward_return_down_0_7": -0.0020474697518158
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 5

- asset: BTC/USDT
- timeframe: 1h
- horizon: 1h
- model: hist_gradient_boosting
- feature_set: crypto_core_v1+crypto_context_proxy_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5575652280796058,
    "metric_pr_auc": 0.5429396708874493,
    "metric_balanced_accuracy": 0.5381617079641091,
    "metric_log_loss": 0.6896754826018888,
    "metric_brier_score": 0.2482568664128242,
    "metric_accuracy": 0.5373287671232877,
    "metric_mean_forward_return": -9.407711287377576e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4974885844748858,
    "metric_signal_coverage_up_0_5": 0.6656392694063927,
    "metric_signal_coverage_down_0_5": 0.3343607305936073,
    "metric_signal_coverage_up_0_55": 0.3067351598173516,
    "metric_signal_coverage_down_0_55": 0.0881278538812785,
    "metric_signal_coverage_up_0_6": 0.0647260273972602,
    "metric_signal_coverage_down_0_6": 0.0084474885844748,
    "metric_signal_coverage_up_0_65": 0.0061643835616438,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0001141552511415,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -5.499527507542512e-05,
    "metric_mean_forward_return_down_0_5": 0.0001718805257116,
    "metric_mean_forward_return_up_0_55": 0.000193386131431,
    "metric_mean_forward_return_down_0_55": 0.0001493150038858,
    "metric_mean_forward_return_up_0_6": 0.0002125023519485,
    "metric_mean_forward_return_down_0_6": 0.0001696128316207,
    "metric_mean_forward_return_up_0_65": 0.0020722437593087,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0010899221944029,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5649516644236572,
    "metric_pr_auc": 0.5708210498400866,
    "metric_balanced_accuracy": 0.5464245833840736,
    "metric_log_loss": 0.6868954447004312,
    "metric_brier_score": 0.2468866787210146,
    "metric_accuracy": 0.5466377440347071,
    "metric_mean_forward_return": 0.0001174623817694,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5134147733759562,
    "metric_signal_coverage_up_0_5": 0.5091905468660806,
    "metric_signal_coverage_down_0_5": 0.4908094531339194,
    "metric_signal_coverage_up_0_55": 0.1633748144765384,
    "metric_signal_coverage_down_0_55": 0.1557255394451421,
    "metric_signal_coverage_up_0_6": 0.0141568672222856,
    "metric_signal_coverage_down_0_6": 0.0073067701792442,
    "metric_signal_coverage_up_0_65": 0.0001141682840506,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001120135812672,
    "metric_mean_forward_return_down_0_5": -0.0001231152429557,
    "metric_mean_forward_return_up_0_55": 0.0003290993256806,
    "metric_mean_forward_return_down_0_55": -0.0004395524653305,
    "metric_mean_forward_return_up_0_6": 0.0002889775273279,
    "metric_mean_forward_return_down_0_6": -0.0015830355468047,
    "metric_mean_forward_return_up_0_65": -0.0018102603411865,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5557226101162673,
    "metric_pr_auc": 0.5589350333559131,
    "metric_balanced_accuracy": 0.5406688194583014,
    "metric_log_loss": 0.6883106684459429,
    "metric_brier_score": 0.2475944869313429,
    "metric_accuracy": 0.5406420765027322,
    "metric_mean_forward_return": 0.0001062213431995,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5114981785063752,
    "metric_signal_coverage_up_0_5": 0.4997723132969034,
    "metric_signal_coverage_down_0_5": 0.5002276867030966,
    "metric_signal_coverage_up_0_55": 0.2258652094717668,
    "metric_signal_coverage_down_0_55": 0.2032103825136612,
    "metric_signal_coverage_up_0_6": 0.043943533697632,
    "metric_signal_coverage_down_0_6": 0.0097905282331511,
    "metric_signal_coverage_up_0_65": 0.0023907103825136,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0002417860089863,
    "metric_mean_forward_return_down_0_5": 2.9219913697047312e-05,
    "metric_mean_forward_return_up_0_55": 0.000471563083401,
    "metric_mean_forward_return_down_0_55": 6.239667405725864e-05,
    "metric_mean_forward_return_up_0_6": 0.0004828374381228,
    "metric_mean_forward_return_down_0_6": 0.0011677501562006,
    "metric_mean_forward_return_up_0_65": 0.0017170673968911,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5377763445659075,
    "metric_pr_auc": 0.533173464814347,
    "metric_balanced_accuracy": 0.5263334770408308,
    "metric_log_loss": 0.6927175263613445,
    "metric_brier_score": 0.2497636963262509,
    "metric_accuracy": 0.5266679917842708,
    "metric_mean_forward_return": 6.073548038943529e-06,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5016895249453389,
    "metric_signal_coverage_up_0_5": 0.5990856688531108,
    "metric_signal_coverage_down_0_5": 0.4009143311468893,
    "metric_signal_coverage_up_0_55": 0.2955012257337839,
    "metric_signal_coverage_down_0_55": 0.13536076326774,
    "metric_signal_coverage_up_0_6": 0.0775856357251706,
    "metric_signal_coverage_down_0_6": 0.0105346849532896,
    "metric_signal_coverage_up_0_65": 0.0049691910157026,
    "metric_signal_coverage_down_0_65": 0.0002650235208374,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 3.333572461217561e-05,
    "metric_mean_forward_return_down_0_5": 3.466428051421538e-05,
    "metric_mean_forward_return_up_0_55": 6.305288743954974e-05,
    "metric_mean_forward_return_down_0_55": 0.0001721207559494,
    "metric_mean_forward_return_up_0_6": 0.0001357256079151,
    "metric_mean_forward_return_down_0_6": 3.527068676641966e-05,
    "metric_mean_forward_return_up_0_65": 0.000626428022512,
    "metric_mean_forward_return_down_0_65": 0.0003713349683575,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 6

- asset: BTC/USDT
- timeframe: 1h
- horizon: 1h
- model: logistic_regression
- feature_set: crypto_core_v1+crypto_context_proxy_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5501430990419265,
    "metric_pr_auc": 0.5377885703193482,
    "metric_balanced_accuracy": 0.5290242096556302,
    "metric_log_loss": 0.6912637186188239,
    "metric_brier_score": 0.2490455426886521,
    "metric_accuracy": 0.5301369863013699,
    "metric_mean_forward_return": -9.407711287377576e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4974885844748858,
    "metric_signal_coverage_up_0_5": 0.278310502283105,
    "metric_signal_coverage_down_0_5": 0.721689497716895,
    "metric_signal_coverage_up_0_55": 0.0662100456621004,
    "metric_signal_coverage_down_0_55": 0.3723744292237443,
    "metric_signal_coverage_up_0_6": 0.0098173515981735,
    "metric_signal_coverage_down_0_6": 0.0909817351598173,
    "metric_signal_coverage_up_0_65": 0.0007990867579908,
    "metric_signal_coverage_down_0_65": 0.0053652968036529,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -5.448155333012176e-05,
    "metric_mean_forward_return_down_0_5": 0.0001093466437449,
    "metric_mean_forward_return_up_0_55": -0.0001391670933573,
    "metric_mean_forward_return_down_0_55": 7.509147670934594e-05,
    "metric_mean_forward_return_up_0_6": 0.0013386751152222,
    "metric_mean_forward_return_down_0_6": 0.0004285994875572,
    "metric_mean_forward_return_up_0_65": 0.0032259302226184,
    "metric_mean_forward_return_down_0_65": -0.0010808141382176,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5575217932973096,
    "metric_pr_auc": 0.5641779252271851,
    "metric_balanced_accuracy": 0.5440997371729231,
    "metric_log_loss": 0.6882440019990641,
    "metric_brier_score": 0.2475611166919655,
    "metric_accuracy": 0.5430985272291358,
    "metric_mean_forward_return": 0.0001174623817694,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5134147733759562,
    "metric_signal_coverage_up_0_5": 0.4638657380979564,
    "metric_signal_coverage_down_0_5": 0.5361342619020436,
    "metric_signal_coverage_up_0_55": 0.126955131864368,
    "metric_signal_coverage_down_0_55": 0.1638314876127412,
    "metric_signal_coverage_up_0_6": 0.0118735015412718,
    "metric_signal_coverage_down_0_6": 0.0089051261559538,
    "metric_signal_coverage_up_0_65": 0.0004566731362027,
    "metric_signal_coverage_down_0_65": 0.0001141682840506,
    "metric_signal_coverage_up_0_7": 0.0001141682840506,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001118129543566,
    "metric_mean_forward_return_down_0_5": -0.00012235029139,
    "metric_mean_forward_return_up_0_55": 9.870041772490164e-05,
    "metric_mean_forward_return_down_0_55": -0.0004807926191119,
    "metric_mean_forward_return_up_0_6": 0.0009983142442853,
    "metric_mean_forward_return_down_0_6": -0.000943734856636,
    "metric_mean_forward_return_up_0_65": 0.0148941434764732,
    "metric_mean_forward_return_down_0_65": 0.0021303749098851,
    "metric_mean_forward_return_up_0_7": 0.0189766692583337,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5452918994683618,
    "metric_pr_auc": 0.5478512726456413,
    "metric_balanced_accuracy": 0.5312471099428444,
    "metric_log_loss": 0.691423953383718,
    "metric_brier_score": 0.2491292342967932,
    "metric_accuracy": 0.5286885245901639,
    "metric_mean_forward_return": 0.0001062213431995,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5114981785063752,
    "metric_signal_coverage_up_0_5": 0.3894581056466302,
    "metric_signal_coverage_down_0_5": 0.6105418943533698,
    "metric_signal_coverage_up_0_55": 0.1226092896174863,
    "metric_signal_coverage_down_0_55": 0.2611566484517304,
    "metric_signal_coverage_up_0_6": 0.0175318761384335,
    "metric_signal_coverage_down_0_6": 0.033811475409836,
    "metric_signal_coverage_up_0_65": 0.0003415300546448,
    "metric_signal_coverage_down_0_65": 0.0009107468123861,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001688720083099,
    "metric_mean_forward_return_down_0_5": -6.625715797810666e-05,
    "metric_mean_forward_return_up_0_55": 0.0004801216023523,
    "metric_mean_forward_return_down_0_55": -0.0001101202791008,
    "metric_mean_forward_return_up_0_6": -0.0001064019847139,
    "metric_mean_forward_return_down_0_6": 0.0005002390737296,
    "metric_mean_forward_return_up_0_65": -0.0028860529411563,
    "metric_mean_forward_return_down_0_65": -0.0013082633789406,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5308838193716161,
    "metric_pr_auc": 0.5254338231904243,
    "metric_balanced_accuracy": 0.5235149031207074,
    "metric_log_loss": 0.6930604767278182,
    "metric_brier_score": 0.2499430567341784,
    "metric_accuracy": 0.5237527330550587,
    "metric_mean_forward_return": 6.073548038943529e-06,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5016895249453389,
    "metric_signal_coverage_up_0_5": 0.5704631286026635,
    "metric_signal_coverage_down_0_5": 0.4295368713973365,
    "metric_signal_coverage_up_0_55": 0.2466706420194792,
    "metric_signal_coverage_down_0_55": 0.1164115815278606,
    "metric_signal_coverage_up_0_6": 0.0373683164380838,
    "metric_signal_coverage_down_0_6": 0.0047041674948651,
    "metric_signal_coverage_up_0_65": 0.0009275823229311,
    "metric_signal_coverage_down_0_65": 0.0002650235208374,
    "metric_signal_coverage_up_0_7": 6.625588020936858e-05,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 1.6460144632476065e-05,
    "metric_mean_forward_return_down_0_5": 7.720775063064053e-06,
    "metric_mean_forward_return_up_0_55": 3.285049257583333e-05,
    "metric_mean_forward_return_down_0_55": 6.153029281571314e-05,
    "metric_mean_forward_return_up_0_6": 1.1916304776694344e-05,
    "metric_mean_forward_return_down_0_6": -0.0001912293283072,
    "metric_mean_forward_return_up_0_65": 0.0034899360800846,
    "metric_mean_forward_return_down_0_65": -0.0086136506511266,
    "metric_mean_forward_return_up_0_7": 0.0202663673823562,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 7

- asset: BTC/USDT
- timeframe: 1h
- horizon: 1h
- model: random_forest
- feature_set: crypto_core_v1+crypto_context_proxy_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5628012549679637,
    "metric_pr_auc": 0.5469544045235452,
    "metric_balanced_accuracy": 0.5412527869700847,
    "metric_log_loss": 0.688396041050132,
    "metric_brier_score": 0.2475514067691217,
    "metric_accuracy": 0.5408675799086758,
    "metric_mean_forward_return": -9.407711287377576e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4974885844748858,
    "metric_signal_coverage_up_0_5": 0.5764840182648402,
    "metric_signal_coverage_down_0_5": 0.4235159817351598,
    "metric_signal_coverage_up_0_55": 0.276027397260274,
    "metric_signal_coverage_down_0_55": 0.1578767123287671,
    "metric_signal_coverage_up_0_6": 0.0738584474885844,
    "metric_signal_coverage_down_0_6": 0.0289954337899543,
    "metric_signal_coverage_up_0_65": 0.0158675799086757,
    "metric_signal_coverage_down_0_65": 0.0014840182648401,
    "metric_signal_coverage_up_0_7": 0.00662100456621,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -6.172292182833211e-05,
    "metric_mean_forward_return_down_0_5": 0.0001381171842429,
    "metric_mean_forward_return_up_0_55": 2.91358877823312e-05,
    "metric_mean_forward_return_down_0_55": 0.0003801202802843,
    "metric_mean_forward_return_up_0_6": 9.560468770564414e-05,
    "metric_mean_forward_return_down_0_6": 0.0003104999698333,
    "metric_mean_forward_return_up_0_65": 0.0008800826096512,
    "metric_mean_forward_return_down_0_65": -0.0002428358268634,
    "metric_mean_forward_return_up_0_7": -0.0020878208434793,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5688808441771547,
    "metric_pr_auc": 0.5731562789568794,
    "metric_balanced_accuracy": 0.548337037246897,
    "metric_log_loss": 0.6862294773477725,
    "metric_brier_score": 0.2465614381797316,
    "metric_accuracy": 0.547436922023062,
    "metric_mean_forward_return": 0.0001174623817694,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5134147733759562,
    "metric_signal_coverage_up_0_5": 0.4677474597556799,
    "metric_signal_coverage_down_0_5": 0.5322525402443201,
    "metric_signal_coverage_up_0_55": 0.1798150473798379,
    "metric_signal_coverage_down_0_55": 0.2146363740152985,
    "metric_signal_coverage_up_0_6": 0.0246603493549491,
    "metric_signal_coverage_down_0_6": 0.0486356890055942,
    "metric_signal_coverage_up_0_65": 0.0022833656810138,
    "metric_signal_coverage_down_0_65": 0.0012558511245575,
    "metric_signal_coverage_up_0_7": 0.0001141682840506,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001124089225545,
    "metric_mean_forward_return_down_0_5": -0.000121903399016,
    "metric_mean_forward_return_up_0_55": 0.0001873551634835,
    "metric_mean_forward_return_down_0_55": -0.0002278299767123,
    "metric_mean_forward_return_up_0_6": 0.0004915178992057,
    "metric_mean_forward_return_down_0_6": -0.0003637331644119,
    "metric_mean_forward_return_up_0_65": 0.002416346759029,
    "metric_mean_forward_return_down_0_65": 0.0013679634919429,
    "metric_mean_forward_return_up_0_7": 0.0111051738035392,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5579351458077437,
    "metric_pr_auc": 0.5604169333178328,
    "metric_balanced_accuracy": 0.5389973258072593,
    "metric_log_loss": 0.6896312918584109,
    "metric_brier_score": 0.2482295566534787,
    "metric_accuracy": 0.5377959927140255,
    "metric_mean_forward_return": 0.0001062213431995,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5114981785063752,
    "metric_signal_coverage_up_0_5": 0.4486566484517304,
    "metric_signal_coverage_down_0_5": 0.5513433515482696,
    "metric_signal_coverage_up_0_55": 0.2106102003642987,
    "metric_signal_coverage_down_0_55": 0.3407331511839708,
    "metric_signal_coverage_up_0_6": 0.043943533697632,
    "metric_signal_coverage_down_0_6": 0.0943761384335154,
    "metric_signal_coverage_up_0_65": 0.0036429872495446,
    "metric_signal_coverage_down_0_65": 0.0013661202185792,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0002304623648588,
    "metric_mean_forward_return_down_0_5": -5.119987354203841e-06,
    "metric_mean_forward_return_up_0_55": 0.000482952605641,
    "metric_mean_forward_return_down_0_55": 5.096994916266625e-05,
    "metric_mean_forward_return_up_0_6": 0.0001767646098728,
    "metric_mean_forward_return_down_0_6": 3.8376510114259705e-05,
    "metric_mean_forward_return_up_0_65": 0.0016297558797601,
    "metric_mean_forward_return_down_0_65": -0.0004290695418099,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5388543351726629,
    "metric_pr_auc": 0.5340222686240107,
    "metric_balanced_accuracy": 0.5267874620897726,
    "metric_log_loss": 0.6925283919670102,
    "metric_brier_score": 0.2496609716599557,
    "metric_accuracy": 0.5269330153051083,
    "metric_mean_forward_return": 6.073548038943529e-06,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5016895249453389,
    "metric_signal_coverage_up_0_5": 0.5431657059564037,
    "metric_signal_coverage_down_0_5": 0.4568342940435964,
    "metric_signal_coverage_up_0_55": 0.2660836149208242,
    "metric_signal_coverage_down_0_55": 0.1782945736434108,
    "metric_signal_coverage_up_0_6": 0.069038627178162,
    "metric_signal_coverage_down_0_6": 0.0123235937189425,
    "metric_signal_coverage_up_0_65": 0.0144437818856423,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0014576293646061,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 2.622673655290916e-05,
    "metric_mean_forward_return_down_0_5": 1.788814005931467e-05,
    "metric_mean_forward_return_up_0_55": 0.0001032000299254,
    "metric_mean_forward_return_down_0_55": 0.0001113320520335,
    "metric_mean_forward_return_up_0_6": 0.0001067042273613,
    "metric_mean_forward_return_down_0_6": 0.0003387393893327,
    "metric_mean_forward_return_up_0_65": 3.072813957725368e-06,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0002207065180415,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 8

- asset: BTC/USDT
- timeframe: 1h
- horizon: 24h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: B
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5285831247685946,
    "metric_pr_auc": 0.4881415174679878,
    "metric_balanced_accuracy": 0.5192257222140091,
    "metric_log_loss": 0.6930272130274996,
    "metric_brier_score": 0.2499171098808345,
    "metric_accuracy": 0.5190639269406393,
    "metric_mean_forward_return": -0.0023088406224894,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4686073059360731,
    "metric_signal_coverage_up_0_5": 0.5013698630136987,
    "metric_signal_coverage_down_0_5": 0.4986301369863014,
    "metric_signal_coverage_up_0_55": 0.0735159817351598,
    "metric_signal_coverage_down_0_55": 0.0424657534246575,
    "metric_signal_coverage_up_0_6": 0.0125570776255707,
    "metric_signal_coverage_down_0_6": 0.0,
    "metric_signal_coverage_up_0_65": 0.0027397260273972,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0003424657534246,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0020794394706297,
    "metric_mean_forward_return_down_0_5": 0.0025395022202385,
    "metric_mean_forward_return_up_0_55": 0.0005517134978708,
    "metric_mean_forward_return_down_0_55": 0.0002679517257063,
    "metric_mean_forward_return_up_0_6": 0.0016682011083021,
    "metric_mean_forward_return_down_0_6": NaN,
    "metric_mean_forward_return_up_0_65": -0.035743554608682,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0070754080963003,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5436618105257873,
    "metric_pr_auc": 0.5710648426771137,
    "metric_balanced_accuracy": 0.5404342235038055,
    "metric_log_loss": 0.6915040393997361,
    "metric_brier_score": 0.2491787865720822,
    "metric_accuracy": 0.5349925790615367,
    "metric_mean_forward_return": 0.0028740545853712,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5274574723141912,
    "metric_signal_coverage_up_0_5": 0.4031282109829889,
    "metric_signal_coverage_down_0_5": 0.5968717890170111,
    "metric_signal_coverage_up_0_55": 0.0078776115994976,
    "metric_signal_coverage_down_0_55": 0.0276287247402671,
    "metric_signal_coverage_up_0_6": 0.0002283365681013,
    "metric_signal_coverage_down_0_6": 0.0001141682840506,
    "metric_signal_coverage_up_0_65": 0.0,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.002244990833397,
    "metric_mean_forward_return_down_0_5": -0.0032989253023224,
    "metric_mean_forward_return_up_0_55": 0.0115182852325767,
    "metric_mean_forward_return_down_0_55": -0.004041549358477,
    "metric_mean_forward_return_up_0_6": 0.0188080389417465,
    "metric_mean_forward_return_down_0_6": -0.0004631605493907,
    "metric_mean_forward_return_up_0_65": NaN,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5403593353796877,
    "metric_pr_auc": 0.5747058077693961,
    "metric_balanced_accuracy": 0.5283568830908848,
    "metric_log_loss": 0.6912135788643097,
    "metric_brier_score": 0.2490370256617123,
    "metric_accuracy": 0.5241347905282332,
    "metric_mean_forward_return": 0.0025133823597573,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5373406193078324,
    "metric_signal_coverage_up_0_5": 0.4455828779599271,
    "metric_signal_coverage_down_0_5": 0.5544171220400729,
    "metric_signal_coverage_up_0_55": 0.0458788706739526,
    "metric_signal_coverage_down_0_55": 0.0631830601092896,
    "metric_signal_coverage_up_0_6": 0.0007969034608378,
    "metric_signal_coverage_down_0_6": 0.0002276867030965,
    "metric_signal_coverage_up_0_65": 0.0,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0028354936189361,
    "metric_mean_forward_return_down_0_5": -0.0022545027974523,
    "metric_mean_forward_return_up_0_55": 0.0071999688847579,
    "metric_mean_forward_return_down_0_55": 0.0007156660990337,
    "metric_mean_forward_return_up_0_6": 0.0710695369687947,
    "metric_mean_forward_return_down_0_6": -0.0324748874707474,
    "metric_mean_forward_return_up_0_65": NaN,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5095865267687495,
    "metric_pr_auc": 0.5245483475689068,
    "metric_balanced_accuracy": 0.5054580925128478,
    "metric_log_loss": 0.6940412728695936,
    "metric_brier_score": 0.2504473812934623,
    "metric_accuracy": 0.5042468480424684,
    "metric_mean_forward_return": 7.211335610647748e-05,
    "metric_sample_count": 15070.0,
    "metric_positive_rate": 0.5112143331121434,
    "metric_signal_coverage_up_0_5": 0.4461181154611811,
    "metric_signal_coverage_down_0_5": 0.5538818845388188,
    "metric_signal_coverage_up_0_55": 0.0354346383543463,
    "metric_signal_coverage_down_0_55": 0.0619774386197743,
    "metric_signal_coverage_up_0_6": 0.0013934970139349,
    "metric_signal_coverage_down_0_6": 6.635700066357e-05,
    "metric_signal_coverage_up_0_65": 6.635700066357e-05,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 8.68418700863252e-06,
    "metric_mean_forward_return_down_0_5": -0.000123201687704,
    "metric_mean_forward_return_up_0_55": 0.0055077610438373,
    "metric_mean_forward_return_down_0_55": 6.615481791951351e-05,
    "metric_mean_forward_return_up_0_6": 0.0413381817189139,
    "metric_mean_forward_return_down_0_6": 0.0225498286233474,
    "metric_mean_forward_return_up_0_65": 0.0805506980339114,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is B; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 9

- asset: BTC/USDT
- timeframe: 1h
- horizon: 2h
- model: hist_gradient_boosting
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5530553793485349,
    "metric_pr_auc": 0.5370599914789926,
    "metric_balanced_accuracy": 0.5424530453395013,
    "metric_log_loss": 0.692215199722228,
    "metric_brier_score": 0.2494464148365261,
    "metric_accuracy": 0.5415525114155251,
    "metric_mean_forward_return": -0.0001892025421138,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4966894977168949,
    "metric_signal_coverage_up_0_5": 0.6357305936073059,
    "metric_signal_coverage_down_0_5": 0.364269406392694,
    "metric_signal_coverage_up_0_55": 0.3608447488584475,
    "metric_signal_coverage_down_0_55": 0.1494292237442922,
    "metric_signal_coverage_up_0_6": 0.129337899543379,
    "metric_signal_coverage_down_0_6": 0.0468036529680365,
    "metric_signal_coverage_up_0_65": 0.0320776255707762,
    "metric_signal_coverage_down_0_65": 0.0130136986301369,
    "metric_signal_coverage_up_0_7": 0.0034246575342465,
    "metric_signal_coverage_down_0_7": 0.0039954337899543,
    "metric_mean_forward_return_up_0_5": -0.0001302979771558,
    "metric_mean_forward_return_down_0_5": 0.0002920040219794,
    "metric_mean_forward_return_up_0_55": -4.069695157101653e-05,
    "metric_mean_forward_return_down_0_55": 2.4680206539806218e-05,
    "metric_mean_forward_return_up_0_6": 8.236042487886532e-05,
    "metric_mean_forward_return_down_0_6": 0.0002976567316454,
    "metric_mean_forward_return_up_0_65": 0.001163382600453,
    "metric_mean_forward_return_down_0_65": 6.623457972694316e-05,
    "metric_mean_forward_return_up_0_7": 0.0032740229305945,
    "metric_mean_forward_return_down_0_7": 0.0003437447193837
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5677950715551692,
    "metric_pr_auc": 0.5697940752675179,
    "metric_balanced_accuracy": 0.5466137158206865,
    "metric_log_loss": 0.6861257997530563,
    "metric_brier_score": 0.2465016021605741,
    "metric_accuracy": 0.5490352779997717,
    "metric_mean_forward_return": 0.000234704560454,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.512729763671652,
    "metric_signal_coverage_up_0_5": 0.5963009475967577,
    "metric_signal_coverage_down_0_5": 0.4036990524032424,
    "metric_signal_coverage_up_0_55": 0.2991209042128097,
    "metric_signal_coverage_down_0_55": 0.1725082772005936,
    "metric_signal_coverage_up_0_6": 0.0927046466491608,
    "metric_signal_coverage_down_0_6": 0.0447539673478707,
    "metric_signal_coverage_up_0_65": 0.0137001940860828,
    "metric_signal_coverage_down_0_65": 0.0038817216577234,
    "metric_signal_coverage_up_0_7": 0.0015983559767096,
    "metric_signal_coverage_down_0_7": 0.0001141682840506,
    "metric_mean_forward_return_up_0_5": 0.0001684387300736,
    "metric_mean_forward_return_down_0_5": -0.0003325853387562,
    "metric_mean_forward_return_up_0_55": 0.0002550928784401,
    "metric_mean_forward_return_down_0_55": -0.0003564857812571,
    "metric_mean_forward_return_up_0_6": 0.00059161872749,
    "metric_mean_forward_return_down_0_6": 0.0001273605381343,
    "metric_mean_forward_return_up_0_65": 0.0005227652168913,
    "metric_mean_forward_return_down_0_65": 0.0012711646383793,
    "metric_mean_forward_return_up_0_7": -0.0002077071571719,
    "metric_mean_forward_return_down_0_7": 0.0011527554216917
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5626201396323818,
    "metric_pr_auc": 0.5704722880746753,
    "metric_balanced_accuracy": 0.5428091544764969,
    "metric_log_loss": 0.6869278938963183,
    "metric_brier_score": 0.2469028394016117,
    "metric_accuracy": 0.5445127504553734,
    "metric_mean_forward_return": 0.0002125073229926,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5170765027322405,
    "metric_signal_coverage_up_0_5": 0.5513433515482696,
    "metric_signal_coverage_down_0_5": 0.4486566484517304,
    "metric_signal_coverage_up_0_55": 0.303620218579235,
    "metric_signal_coverage_down_0_55": 0.2007058287795992,
    "metric_signal_coverage_up_0_6": 0.0906193078324225,
    "metric_signal_coverage_down_0_6": 0.0400728597449908,
    "metric_signal_coverage_up_0_65": 0.0125227686703096,
    "metric_signal_coverage_down_0_65": 0.0058060109289617,
    "metric_signal_coverage_up_0_7": 0.0005692167577413,
    "metric_signal_coverage_down_0_7": 0.0002276867030965,
    "metric_mean_forward_return_up_0_5": 0.0003680884380559,
    "metric_mean_forward_return_down_0_5": -2.1317437112977903e-05,
    "metric_mean_forward_return_up_0_55": 0.0006525318152956,
    "metric_mean_forward_return_down_0_55": 0.0003753279849169,
    "metric_mean_forward_return_up_0_6": 0.001139394392364,
    "metric_mean_forward_return_down_0_6": -0.000321456561344,
    "metric_mean_forward_return_up_0_65": 0.0019962042136424,
    "metric_mean_forward_return_down_0_65": -5.2312351437424806e-05,
    "metric_mean_forward_return_up_0_7": 0.0036602162691182,
    "metric_mean_forward_return_down_0_7": 0.008304832076311
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.535891675472939,
    "metric_pr_auc": 0.5381454422638197,
    "metric_balanced_accuracy": 0.5281226901100878,
    "metric_log_loss": 0.6944637483259098,
    "metric_brier_score": 0.2505944163018177,
    "metric_accuracy": 0.5290882586800955,
    "metric_mean_forward_return": 1.158874812000908e-05,
    "metric_sample_count": 15092.0,
    "metric_positive_rate": 0.507553670818977,
    "metric_signal_coverage_up_0_5": 0.5643387225019878,
    "metric_signal_coverage_down_0_5": 0.4356612774980122,
    "metric_signal_coverage_up_0_55": 0.3118208322289955,
    "metric_signal_coverage_down_0_55": 0.2065332626557116,
    "metric_signal_coverage_up_0_6": 0.1132388020143122,
    "metric_signal_coverage_down_0_6": 0.0472435727537768,
    "metric_signal_coverage_up_0_65": 0.020474423535648,
    "metric_signal_coverage_down_0_65": 0.0070898489265836,
    "metric_signal_coverage_up_0_7": 0.0003313013517095,
    "metric_signal_coverage_down_0_7": 0.0012589451364961,
    "metric_mean_forward_return_up_0_5": 0.0001008493391714,
    "metric_mean_forward_return_down_0_5": 0.0001040359597103,
    "metric_mean_forward_return_up_0_55": 5.766265445437813e-05,
    "metric_mean_forward_return_down_0_55": 8.284229720239829e-06,
    "metric_mean_forward_return_up_0_6": -4.309680479681672e-05,
    "metric_mean_forward_return_down_0_6": -4.210054905433173e-05,
    "metric_mean_forward_return_up_0_65": 0.0003230406588316,
    "metric_mean_forward_return_down_0_65": 0.0004729612028941,
    "metric_mean_forward_return_up_0_7": 0.0115195599544287,
    "metric_mean_forward_return_down_0_7": 0.000558157274994
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 10

- asset: BTC/USDT
- timeframe: 1h
- horizon: 2h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5462058943285758,
    "metric_pr_auc": 0.5285363260160072,
    "metric_balanced_accuracy": 0.5357956258273036,
    "metric_log_loss": 0.6909123469244267,
    "metric_brier_score": 0.2488626006829555,
    "metric_accuracy": 0.5356164383561643,
    "metric_mean_forward_return": -0.0001892025421138,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4966894977168949,
    "metric_signal_coverage_up_0_5": 0.5268264840182648,
    "metric_signal_coverage_down_0_5": 0.4731735159817352,
    "metric_signal_coverage_up_0_55": 0.1837899543378995,
    "metric_signal_coverage_down_0_55": 0.1431506849315068,
    "metric_signal_coverage_up_0_6": 0.0355022831050228,
    "metric_signal_coverage_down_0_6": 0.0182648401826484,
    "metric_signal_coverage_up_0_65": 0.0042237442922374,
    "metric_signal_coverage_down_0_65": 0.0009132420091324,
    "metric_signal_coverage_up_0_7": 0.0003424657534246,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0001363852462925,
    "metric_mean_forward_return_down_0_5": 0.0002480087713575,
    "metric_mean_forward_return_up_0_55": 8.479395126895938e-05,
    "metric_mean_forward_return_down_0_55": 0.0003132194221589,
    "metric_mean_forward_return_up_0_6": -0.0015627566967552,
    "metric_mean_forward_return_down_0_6": 0.0013714921999289,
    "metric_mean_forward_return_up_0_65": 0.0035844037105271,
    "metric_mean_forward_return_down_0_65": 0.0050194828878286,
    "metric_mean_forward_return_up_0_7": 0.0464063758198324,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5638978154163163,
    "metric_pr_auc": 0.5741624549156992,
    "metric_balanced_accuracy": 0.5491906962941816,
    "metric_log_loss": 0.6872046802100512,
    "metric_brier_score": 0.2470464875267552,
    "metric_accuracy": 0.5490352779997717,
    "metric_mean_forward_return": 0.000234704560454,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.512729763671652,
    "metric_signal_coverage_up_0_5": 0.4951478479278456,
    "metric_signal_coverage_down_0_5": 0.5048521520721544,
    "metric_signal_coverage_up_0_55": 0.1264984587281653,
    "metric_signal_coverage_down_0_55": 0.1270693001484187,
    "metric_signal_coverage_up_0_6": 0.0117593332572211,
    "metric_signal_coverage_down_0_6": 0.0053659093503824,
    "metric_signal_coverage_up_0_65": 0.0012558511245575,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0001141682840506,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001966534004042,
    "metric_mean_forward_return_down_0_5": -0.0002720242983861,
    "metric_mean_forward_return_up_0_55": 0.0003462412601324,
    "metric_mean_forward_return_down_0_55": -0.0006658204515567,
    "metric_mean_forward_return_up_0_6": 0.0012318878735657,
    "metric_mean_forward_return_down_0_6": -0.0005849041548055,
    "metric_mean_forward_return_up_0_65": 0.0066076053833944,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0284537071836994,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5451669482856948,
    "metric_pr_auc": 0.556584147992278,
    "metric_balanced_accuracy": 0.5323161727382401,
    "metric_log_loss": 0.6902527044746745,
    "metric_brier_score": 0.2485562048385323,
    "metric_accuracy": 0.530851548269581,
    "metric_mean_forward_return": 0.0002125073229926,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5170765027322405,
    "metric_signal_coverage_up_0_5": 0.4582194899817851,
    "metric_signal_coverage_down_0_5": 0.5417805100182149,
    "metric_signal_coverage_up_0_55": 0.1602914389799635,
    "metric_signal_coverage_down_0_55": 0.1734972677595628,
    "metric_signal_coverage_up_0_6": 0.0299408014571949,
    "metric_signal_coverage_down_0_6": 0.0113843351548269,
    "metric_signal_coverage_up_0_65": 0.001707650273224,
    "metric_signal_coverage_down_0_65": 0.0002276867030965,
    "metric_signal_coverage_up_0_7": 0.0001138433515482,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0002652980596388,
    "metric_mean_forward_return_down_0_5": -0.0001678587171928,
    "metric_mean_forward_return_up_0_55": 0.000618257807318,
    "metric_mean_forward_return_down_0_55": 1.493981061724646e-05,
    "metric_mean_forward_return_up_0_6": 0.0010243633168291,
    "metric_mean_forward_return_down_0_6": 0.0010187837606663,
    "metric_mean_forward_return_up_0_65": 0.0027192357520686,
    "metric_mean_forward_return_down_0_65": 9.539571731853602e-05,
    "metric_mean_forward_return_up_0_7": 0.0400776368326527,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5323634372005048,
    "metric_pr_auc": 0.5311864556233317,
    "metric_balanced_accuracy": 0.5216326547819463,
    "metric_log_loss": 0.692612316789636,
    "metric_brier_score": 0.2497244143196778,
    "metric_accuracy": 0.521203286509409,
    "metric_mean_forward_return": 1.158874812000908e-05,
    "metric_sample_count": 15092.0,
    "metric_positive_rate": 0.507553670818977,
    "metric_signal_coverage_up_0_5": 0.4719056453750331,
    "metric_signal_coverage_down_0_5": 0.5280943546249669,
    "metric_signal_coverage_up_0_55": 0.1675722236946726,
    "metric_signal_coverage_down_0_55": 0.181685661277498,
    "metric_signal_coverage_up_0_6": 0.0175589716406042,
    "metric_signal_coverage_down_0_6": 0.0076861913596607,
    "metric_signal_coverage_up_0_65": 0.0007951232441028,
    "metric_signal_coverage_down_0_65": 0.0001325205406838,
    "metric_signal_coverage_up_0_7": 6.626027034190299e-05,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 1.4312310607625184e-05,
    "metric_mean_forward_return_down_0_5": -9.154969947261038e-06,
    "metric_mean_forward_return_up_0_55": -0.0001936150541997,
    "metric_mean_forward_return_down_0_55": 0.000326850703884,
    "metric_mean_forward_return_up_0_6": -0.0004960823271809,
    "metric_mean_forward_return_down_0_6": 0.0010853729557137,
    "metric_mean_forward_return_up_0_65": 0.0075359814873087,
    "metric_mean_forward_return_down_0_65": 0.0078904385979136,
    "metric_mean_forward_return_up_0_7": 0.0217501510117401,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 11

- asset: BTC/USDT
- timeframe: 1h
- horizon: 2h
- model: random_forest
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5563224217153866,
    "metric_pr_auc": 0.5406573251051479,
    "metric_balanced_accuracy": 0.53957876116731,
    "metric_log_loss": 0.6898736025206818,
    "metric_brier_score": 0.2483155981913784,
    "metric_accuracy": 0.539269406392694,
    "metric_mean_forward_return": -0.0001892025421138,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4966894977168949,
    "metric_signal_coverage_up_0_5": 0.5464611872146119,
    "metric_signal_coverage_down_0_5": 0.4535388127853881,
    "metric_signal_coverage_up_0_55": 0.2801369863013698,
    "metric_signal_coverage_down_0_55": 0.2246575342465753,
    "metric_signal_coverage_up_0_6": 0.0742009132420091,
    "metric_signal_coverage_down_0_6": 0.0643835616438356,
    "metric_signal_coverage_up_0_65": 0.0146118721461187,
    "metric_signal_coverage_down_0_65": 0.0114155251141552,
    "metric_signal_coverage_up_0_7": 0.0031963470319634,
    "metric_signal_coverage_down_0_7": 0.0011415525114155,
    "metric_mean_forward_return_up_0_5": -0.0001731651667394,
    "metric_mean_forward_return_down_0_5": 0.0002085257024251,
    "metric_mean_forward_return_up_0_55": 0.0001055225339902,
    "metric_mean_forward_return_down_0_55": 0.0001419578066785,
    "metric_mean_forward_return_up_0_6": 0.0004303149496041,
    "metric_mean_forward_return_down_0_6": -0.0002339786958658,
    "metric_mean_forward_return_up_0_65": 0.0012579540917267,
    "metric_mean_forward_return_down_0_65": 0.0001415093800525,
    "metric_mean_forward_return_up_0_7": 0.0031686228332058,
    "metric_mean_forward_return_down_0_7": -0.0022447122940369
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5710611058626678,
    "metric_pr_auc": 0.5726139052416106,
    "metric_balanced_accuracy": 0.5519772753880143,
    "metric_log_loss": 0.6857801633442616,
    "metric_brier_score": 0.246327359995243,
    "metric_accuracy": 0.5523461582372416,
    "metric_mean_forward_return": 0.000234704560454,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.512729763671652,
    "metric_signal_coverage_up_0_5": 0.5158123073410207,
    "metric_signal_coverage_down_0_5": 0.4841876926589793,
    "metric_signal_coverage_up_0_55": 0.2306199337823952,
    "metric_signal_coverage_down_0_55": 0.2208014613540358,
    "metric_signal_coverage_up_0_6": 0.0521749058111656,
    "metric_signal_coverage_down_0_6": 0.0486356890055942,
    "metric_signal_coverage_up_0_65": 0.00662176047494,
    "metric_signal_coverage_down_0_65": 0.005137572782281,
    "metric_signal_coverage_up_0_7": 0.0013700194086082,
    "metric_signal_coverage_down_0_7": 0.0001141682840506,
    "metric_mean_forward_return_up_0_5": 0.0001878138523823,
    "metric_mean_forward_return_down_0_5": -0.0002846579250066,
    "metric_mean_forward_return_up_0_55": 0.0003348628115958,
    "metric_mean_forward_return_down_0_55": -0.0003428500648892,
    "metric_mean_forward_return_up_0_6": 0.0008667817865738,
    "metric_mean_forward_return_down_0_6": -0.000447633816502,
    "metric_mean_forward_return_up_0_65": 0.0016473724418659,
    "metric_mean_forward_return_down_0_65": -0.0010547578134922,
    "metric_mean_forward_return_up_0_7": -0.0023163808886228,
    "metric_mean_forward_return_down_0_7": 0.0011931318469498
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5633321541250181,
    "metric_pr_auc": 0.5700181348253165,
    "metric_balanced_accuracy": 0.5443674533522422,
    "metric_log_loss": 0.6876280877117295,
    "metric_brier_score": 0.2472444399105669,
    "metric_accuracy": 0.5438296903460837,
    "metric_mean_forward_return": 0.0002125073229926,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5170765027322405,
    "metric_signal_coverage_up_0_5": 0.4857695810564663,
    "metric_signal_coverage_down_0_5": 0.5142304189435337,
    "metric_signal_coverage_up_0_55": 0.2637750455373406,
    "metric_signal_coverage_down_0_55": 0.2784608378870674,
    "metric_signal_coverage_up_0_6": 0.0704690346083788,
    "metric_signal_coverage_down_0_6": 0.0703551912568306,
    "metric_signal_coverage_up_0_65": 0.0095628415300546,
    "metric_signal_coverage_down_0_65": 0.0033014571948998,
    "metric_signal_coverage_up_0_7": 0.0011384335154826,
    "metric_signal_coverage_down_0_7": 0.0003415300546448,
    "metric_mean_forward_return_up_0_5": 0.0004306916082831,
    "metric_mean_forward_return_down_0_5": -6.398767461335275e-06,
    "metric_mean_forward_return_up_0_55": 0.0005889126037817,
    "metric_mean_forward_return_down_0_55": 0.0002092932899639,
    "metric_mean_forward_return_up_0_6": 0.0010833386002644,
    "metric_mean_forward_return_down_0_6": -0.0001398534571211,
    "metric_mean_forward_return_up_0_65": 0.0022356939210219,
    "metric_mean_forward_return_down_0_65": 0.0004524126078452,
    "metric_mean_forward_return_up_0_7": 0.0090107331694041,
    "metric_mean_forward_return_down_0_7": 0.0006009606658781
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5355708115635723,
    "metric_pr_auc": 0.5363089125310627,
    "metric_balanced_accuracy": 0.5254898020556088,
    "metric_log_loss": 0.6944011042731453,
    "metric_brier_score": 0.2505777389221404,
    "metric_accuracy": 0.5254439438112908,
    "metric_mean_forward_return": 1.158874812000908e-05,
    "metric_sample_count": 15092.0,
    "metric_positive_rate": 0.507553670818977,
    "metric_signal_coverage_up_0_5": 0.4973495891863239,
    "metric_signal_coverage_down_0_5": 0.5026504108136761,
    "metric_signal_coverage_up_0_55": 0.2682878346143652,
    "metric_signal_coverage_down_0_55": 0.2814736284124039,
    "metric_signal_coverage_up_0_6": 0.0738802014312218,
    "metric_signal_coverage_down_0_6": 0.0722899549430161,
    "metric_signal_coverage_up_0_65": 0.0121918897429101,
    "metric_signal_coverage_down_0_65": 0.0056983832494036,
    "metric_signal_coverage_up_0_7": 0.0011264245958123,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 4.817471268148555e-05,
    "metric_mean_forward_return_down_0_5": 2.4611390292651403e-05,
    "metric_mean_forward_return_up_0_55": 5.258708663129426e-05,
    "metric_mean_forward_return_down_0_55": 0.000139136052753,
    "metric_mean_forward_return_up_0_6": -0.0001376516335367,
    "metric_mean_forward_return_down_0_6": -0.0001799561183772,
    "metric_mean_forward_return_up_0_65": 4.3618472303032535e-05,
    "metric_mean_forward_return_down_0_65": -0.0001487120706677,
    "metric_mean_forward_return_up_0_7": 0.0010326123211926,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 12

- asset: BTC/USDT
- timeframe: 1h
- horizon: 2h
- model: random_forest
- feature_set: crypto_core_v1+crypto_context_proxy_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5481707539252753,
    "metric_pr_auc": 0.531713717365568,
    "metric_balanced_accuracy": 0.5362687392886795,
    "metric_log_loss": 0.6918929981093417,
    "metric_brier_score": 0.2493264698962306,
    "metric_accuracy": 0.5353881278538812,
    "metric_mean_forward_return": -0.0001892025421138,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4966894977168949,
    "metric_signal_coverage_up_0_5": 0.6327625570776255,
    "metric_signal_coverage_down_0_5": 0.3672374429223744,
    "metric_signal_coverage_up_0_55": 0.3376712328767123,
    "metric_signal_coverage_down_0_55": 0.1494292237442922,
    "metric_signal_coverage_up_0_6": 0.1050228310502283,
    "metric_signal_coverage_down_0_6": 0.0299086757990867,
    "metric_signal_coverage_up_0_65": 0.0148401826484018,
    "metric_signal_coverage_down_0_65": 0.0014840182648401,
    "metric_signal_coverage_up_0_7": 0.0025114155251141,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0001525679101821,
    "metric_mean_forward_return_down_0_5": 0.0002523252542049,
    "metric_mean_forward_return_up_0_55": -7.667974692103154e-05,
    "metric_mean_forward_return_down_0_55": 0.0003332725333011,
    "metric_mean_forward_return_up_0_6": 0.0002198202218314,
    "metric_mean_forward_return_down_0_6": 8.672650668270817e-08,
    "metric_mean_forward_return_up_0_65": 0.0011782170032931,
    "metric_mean_forward_return_down_0_65": 9.211909383035109e-05,
    "metric_mean_forward_return_up_0_7": 0.0051177332765584,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5751760732753646,
    "metric_pr_auc": 0.579649737020536,
    "metric_balanced_accuracy": 0.5554845241873938,
    "metric_log_loss": 0.6850976897141661,
    "metric_brier_score": 0.2459921097792911,
    "metric_accuracy": 0.5547436922023062,
    "metric_mean_forward_return": 0.000234704560454,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.512729763671652,
    "metric_signal_coverage_up_0_5": 0.4723141911177075,
    "metric_signal_coverage_down_0_5": 0.5276858088822926,
    "metric_signal_coverage_up_0_55": 0.1961411119990866,
    "metric_signal_coverage_down_0_55": 0.2238840050234045,
    "metric_signal_coverage_up_0_6": 0.0304829318415344,
    "metric_signal_coverage_down_0_6": 0.0381322068729307,
    "metric_signal_coverage_up_0_65": 0.0014841876926589,
    "metric_signal_coverage_down_0_65": 0.0012558511245575,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.000249055246864,
    "metric_mean_forward_return_down_0_5": -0.0002218597336088,
    "metric_mean_forward_return_up_0_55": 0.0004195958932612,
    "metric_mean_forward_return_down_0_55": -0.0004104117675343,
    "metric_mean_forward_return_up_0_6": 0.000730371926589,
    "metric_mean_forward_return_down_0_6": -0.0010965182681459,
    "metric_mean_forward_return_up_0_65": 0.0020706740405442,
    "metric_mean_forward_return_down_0_65": 0.0003121769235722,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5586603716042486,
    "metric_pr_auc": 0.5678437654745142,
    "metric_balanced_accuracy": 0.5400302815712785,
    "metric_log_loss": 0.6904425279940974,
    "metric_brier_score": 0.2486387817099516,
    "metric_accuracy": 0.5364298724954463,
    "metric_mean_forward_return": 0.0002125073229926,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5170765027322405,
    "metric_signal_coverage_up_0_5": 0.3959471766848816,
    "metric_signal_coverage_down_0_5": 0.6040528233151184,
    "metric_signal_coverage_up_0_55": 0.1977459016393442,
    "metric_signal_coverage_down_0_55": 0.3854735883424408,
    "metric_signal_coverage_up_0_6": 0.0593123861566484,
    "metric_signal_coverage_down_0_6": 0.1239754098360655,
    "metric_signal_coverage_up_0_65": 0.007627504553734,
    "metric_signal_coverage_down_0_65": 0.004667577413479,
    "metric_signal_coverage_up_0_7": 0.0002276867030965,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0003735601649507,
    "metric_mean_forward_return_down_0_5": -0.0001069397043852,
    "metric_mean_forward_return_up_0_55": 0.0005318416719297,
    "metric_mean_forward_return_down_0_55": 4.012641814338184e-06,
    "metric_mean_forward_return_up_0_6": 0.0008381406782208,
    "metric_mean_forward_return_down_0_6": 0.0001652396145019,
    "metric_mean_forward_return_up_0_65": 0.0031313921849728,
    "metric_mean_forward_return_down_0_65": 0.000797559453472,
    "metric_mean_forward_return_up_0_7": -0.0036626658233816,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5389806833479949,
    "metric_pr_auc": 0.5391386517551551,
    "metric_balanced_accuracy": 0.5298739204119087,
    "metric_log_loss": 0.6924567803578733,
    "metric_brier_score": 0.2496290326153026,
    "metric_accuracy": 0.5304797243572754,
    "metric_mean_forward_return": 1.158874812000908e-05,
    "metric_sample_count": 15092.0,
    "metric_positive_rate": 0.507553670818977,
    "metric_signal_coverage_up_0_5": 0.5405512854492446,
    "metric_signal_coverage_down_0_5": 0.4594487145507553,
    "metric_signal_coverage_up_0_55": 0.2747813411078717,
    "metric_signal_coverage_down_0_55": 0.1963291810230585,
    "metric_signal_coverage_up_0_6": 0.0765968725152398,
    "metric_signal_coverage_down_0_6": 0.0210707659687251,
    "metric_signal_coverage_up_0_65": 0.008878876225815,
    "metric_signal_coverage_down_0_65": 0.0003313013517095,
    "metric_signal_coverage_up_0_7": 0.0001325205406838,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 7.132378363127178e-05,
    "metric_mean_forward_return_down_0_5": 5.8690804764456025e-05,
    "metric_mean_forward_return_up_0_55": 3.095491424373402e-05,
    "metric_mean_forward_return_down_0_55": 0.0001276815230679,
    "metric_mean_forward_return_up_0_6": 4.699743309202417e-05,
    "metric_mean_forward_return_down_0_6": -0.0010602778755692,
    "metric_mean_forward_return_up_0_65": -0.0008710962536843,
    "metric_mean_forward_return_down_0_65": -0.0009652810847443,
    "metric_mean_forward_return_up_0_7": -0.0031470594024449,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 13

- asset: BTC/USDT
- timeframe: 1h
- horizon: 4h
- model: hist_gradient_boosting
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5481512841282477,
    "metric_pr_auc": 0.5330818529892649,
    "metric_balanced_accuracy": 0.5330465269765636,
    "metric_log_loss": 0.6932341093511747,
    "metric_brier_score": 0.2499399039671392,
    "metric_accuracy": 0.5320776255707762,
    "metric_mean_forward_return": -0.0003814488729362,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4966894977168949,
    "metric_signal_coverage_up_0_5": 0.6461187214611872,
    "metric_signal_coverage_down_0_5": 0.3538812785388128,
    "metric_signal_coverage_up_0_55": 0.3781963470319635,
    "metric_signal_coverage_down_0_55": 0.1374429223744292,
    "metric_signal_coverage_up_0_6": 0.1257990867579908,
    "metric_signal_coverage_down_0_6": 0.0320776255707762,
    "metric_signal_coverage_up_0_65": 0.030593607305936,
    "metric_signal_coverage_down_0_65": 0.0082191780821917,
    "metric_signal_coverage_up_0_7": 0.0058219178082191,
    "metric_signal_coverage_down_0_7": 0.0023972602739726,
    "metric_mean_forward_return_up_0_5": -0.0004110260498456,
    "metric_mean_forward_return_down_0_5": 0.0003274466725143,
    "metric_mean_forward_return_up_0_55": -0.0003936054285686,
    "metric_mean_forward_return_down_0_55": 0.0005462366749187,
    "metric_mean_forward_return_up_0_6": -0.0006202194842766,
    "metric_mean_forward_return_down_0_6": 0.0013356608169529,
    "metric_mean_forward_return_up_0_65": -0.0003117805699359,
    "metric_mean_forward_return_down_0_65": 0.001333597692939,
    "metric_mean_forward_return_up_0_7": -0.0024625032435415,
    "metric_mean_forward_return_down_0_7": 0.0018376997615155
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5666212558787074,
    "metric_pr_auc": 0.5575567586170815,
    "metric_balanced_accuracy": 0.5480787434641479,
    "metric_log_loss": 0.687098792472103,
    "metric_brier_score": 0.2469637236689434,
    "metric_accuracy": 0.5498344559881265,
    "metric_mean_forward_return": 0.0004712740487291,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5089622102979792,
    "metric_signal_coverage_up_0_5": 0.5988126498458728,
    "metric_signal_coverage_down_0_5": 0.4011873501541272,
    "metric_signal_coverage_up_0_55": 0.3165886516725653,
    "metric_signal_coverage_down_0_55": 0.1494462838223541,
    "metric_signal_coverage_up_0_6": 0.0867678958785249,
    "metric_signal_coverage_down_0_6": 0.0271720516040643,
    "metric_signal_coverage_up_0_65": 0.0142710355063363,
    "metric_signal_coverage_down_0_65": 0.0022833656810138,
    "metric_signal_coverage_up_0_7": 0.0010275145564562,
    "metric_signal_coverage_down_0_7": 0.0001141682840506,
    "metric_mean_forward_return_up_0_5": 0.0004912868821987,
    "metric_mean_forward_return_down_0_5": -0.0004414028729899,
    "metric_mean_forward_return_up_0_55": 0.0005992014733318,
    "metric_mean_forward_return_down_0_55": -0.0002527956953627,
    "metric_mean_forward_return_up_0_6": 0.0009704212603578,
    "metric_mean_forward_return_down_0_6": -0.0002212881181129,
    "metric_mean_forward_return_up_0_65": 0.0002985882724026,
    "metric_mean_forward_return_down_0_65": 0.0018543797122827,
    "metric_mean_forward_return_up_0_7": 0.0019810280557802,
    "metric_mean_forward_return_down_0_7": 0.0025074247230632
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5576286484284962,
    "metric_pr_auc": 0.5721852494010151,
    "metric_balanced_accuracy": 0.5370915212882154,
    "metric_log_loss": 0.6875316768784574,
    "metric_brier_score": 0.2472138666827033,
    "metric_accuracy": 0.5391621129326047,
    "metric_mean_forward_return": 0.0004221157517958,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.519808743169399,
    "metric_signal_coverage_up_0_5": 0.5537340619307832,
    "metric_signal_coverage_down_0_5": 0.4462659380692167,
    "metric_signal_coverage_up_0_55": 0.3084016393442623,
    "metric_signal_coverage_down_0_55": 0.1887522768670309,
    "metric_signal_coverage_up_0_6": 0.0809426229508196,
    "metric_signal_coverage_down_0_6": 0.0231102003642987,
    "metric_signal_coverage_up_0_65": 0.0103597449908925,
    "metric_signal_coverage_down_0_65": 0.0009107468123861,
    "metric_signal_coverage_up_0_7": 0.0007969034608378,
    "metric_signal_coverage_down_0_7": 0.0001138433515482,
    "metric_mean_forward_return_up_0_5": 0.0005734542036078,
    "metric_mean_forward_return_down_0_5": -0.0002343325299557,
    "metric_mean_forward_return_up_0_55": 0.0008795789302326,
    "metric_mean_forward_return_down_0_55": 3.909787959638869e-05,
    "metric_mean_forward_return_up_0_6": 0.001122263817549,
    "metric_mean_forward_return_down_0_6": -0.0015031895997618,
    "metric_mean_forward_return_up_0_65": 0.0024678614854084,
    "metric_mean_forward_return_down_0_65": -0.0057770604997778,
    "metric_mean_forward_return_up_0_7": 0.0056200593663169,
    "metric_mean_forward_return_down_0_7": 0.0032832084426277
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5341629287381664,
    "metric_pr_auc": 0.5273392959011202,
    "metric_balanced_accuracy": 0.5263508257992828,
    "metric_log_loss": 0.6944497728927445,
    "metric_brier_score": 0.2505880239505234,
    "metric_accuracy": 0.5270377733598409,
    "metric_mean_forward_return": 2.2535562542237453e-05,
    "metric_sample_count": 15090.0,
    "metric_positive_rate": 0.5046388336646785,
    "metric_signal_coverage_up_0_5": 0.5742876076872101,
    "metric_signal_coverage_down_0_5": 0.4257123923127899,
    "metric_signal_coverage_up_0_55": 0.3190192180251822,
    "metric_signal_coverage_down_0_55": 0.1834990059642147,
    "metric_signal_coverage_up_0_6": 0.1,
    "metric_signal_coverage_down_0_6": 0.0311464546056991,
    "metric_signal_coverage_up_0_65": 0.0176275679257786,
    "metric_signal_coverage_down_0_65": 0.0024519549370444,
    "metric_signal_coverage_up_0_7": 0.0015241882041086,
    "metric_signal_coverage_down_0_7": 0.0001988071570576,
    "metric_mean_forward_return_up_0_5": 2.1791766528193148e-05,
    "metric_mean_forward_return_down_0_5": -2.3538946143997737e-05,
    "metric_mean_forward_return_up_0_55": -8.431173978592706e-06,
    "metric_mean_forward_return_down_0_55": -4.703487400019088e-05,
    "metric_mean_forward_return_up_0_6": -0.0003543607360159,
    "metric_mean_forward_return_down_0_6": -0.0002983929931946,
    "metric_mean_forward_return_up_0_65": 0.0001912853999496,
    "metric_mean_forward_return_down_0_65": -0.0003476179627924,
    "metric_mean_forward_return_up_0_7": 0.0009676076591345,
    "metric_mean_forward_return_down_0_7": -0.0010995829595505
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 14

- asset: BTC/USDT
- timeframe: 1h
- horizon: 4h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5402524109316733,
    "metric_pr_auc": 0.5262636597255644,
    "metric_balanced_accuracy": 0.5329226969823483,
    "metric_log_loss": 0.6914987882839378,
    "metric_brier_score": 0.2491633770432564,
    "metric_accuracy": 0.532648401826484,
    "metric_mean_forward_return": -0.0003814488729362,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4966894977168949,
    "metric_signal_coverage_up_0_5": 0.5412100456621004,
    "metric_signal_coverage_down_0_5": 0.4587899543378995,
    "metric_signal_coverage_up_0_55": 0.1884703196347032,
    "metric_signal_coverage_down_0_55": 0.1339041095890411,
    "metric_signal_coverage_up_0_6": 0.0297945205479452,
    "metric_signal_coverage_down_0_6": 0.0127853881278538,
    "metric_signal_coverage_up_0_65": 0.0026255707762557,
    "metric_signal_coverage_down_0_65": 0.0004566210045662,
    "metric_signal_coverage_up_0_7": 0.0004566210045662,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0002710754398925,
    "metric_mean_forward_return_down_0_5": 0.0005116505265962,
    "metric_mean_forward_return_up_0_55": -0.0002343292774013,
    "metric_mean_forward_return_down_0_55": 0.0009923108007858,
    "metric_mean_forward_return_up_0_6": -0.0005801886821843,
    "metric_mean_forward_return_down_0_6": 0.0022600452971264,
    "metric_mean_forward_return_up_0_65": -0.0009664600631898,
    "metric_mean_forward_return_down_0_65": 0.011468692167336,
    "metric_mean_forward_return_up_0_7": 0.0253081720980732,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5668824709143042,
    "metric_pr_auc": 0.5695150875777295,
    "metric_balanced_accuracy": 0.5571793636940463,
    "metric_log_loss": 0.6872036204485461,
    "metric_brier_score": 0.2470382882154009,
    "metric_accuracy": 0.5575978993035735,
    "metric_mean_forward_return": 0.0004712740487291,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5089622102979792,
    "metric_signal_coverage_up_0_5": 0.5243749286448225,
    "metric_signal_coverage_down_0_5": 0.4756250713551775,
    "metric_signal_coverage_up_0_55": 0.1227309053544925,
    "metric_signal_coverage_down_0_55": 0.1065190090192944,
    "metric_signal_coverage_up_0_6": 0.0055942459184838,
    "metric_signal_coverage_down_0_6": 0.0021691973969631,
    "metric_signal_coverage_up_0_65": 0.0002283365681013,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0004791393453505,
    "metric_mean_forward_return_down_0_5": -0.0004626025875237,
    "metric_mean_forward_return_up_0_55": 0.0008676403085284,
    "metric_mean_forward_return_down_0_55": -0.0013671965046191,
    "metric_mean_forward_return_up_0_6": 0.0035713754303663,
    "metric_mean_forward_return_down_0_6": -0.0080186423883162,
    "metric_mean_forward_return_up_0_65": 0.0139586776581752,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5426864550420814,
    "metric_pr_auc": 0.5588131609612059,
    "metric_balanced_accuracy": 0.5299814822776301,
    "metric_log_loss": 0.690203944052208,
    "metric_brier_score": 0.2485386547594516,
    "metric_accuracy": 0.5288023679417122,
    "metric_mean_forward_return": 0.0004221157517958,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.519808743169399,
    "metric_signal_coverage_up_0_5": 0.4714253187613843,
    "metric_signal_coverage_down_0_5": 0.5285746812386156,
    "metric_signal_coverage_up_0_55": 0.1505009107468123,
    "metric_signal_coverage_down_0_55": 0.1563069216757741,
    "metric_signal_coverage_up_0_6": 0.0173041894353369,
    "metric_signal_coverage_down_0_6": 0.0080828779599271,
    "metric_signal_coverage_up_0_65": 0.0002276867030965,
    "metric_signal_coverage_down_0_65": 0.0003415300546448,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.000424187398837,
    "metric_mean_forward_return_down_0_5": -0.0004202680907152,
    "metric_mean_forward_return_up_0_55": 0.0009907787522947,
    "metric_mean_forward_return_down_0_55": 0.0001733120547961,
    "metric_mean_forward_return_up_0_6": 0.0028623362409614,
    "metric_mean_forward_return_down_0_6": 0.0037452272217808,
    "metric_mean_forward_return_up_0_65": 0.0236323140864704,
    "metric_mean_forward_return_down_0_65": 0.0034872067962098,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5298004422709096,
    "metric_pr_auc": 0.5232028285750684,
    "metric_balanced_accuracy": 0.5181117184223183,
    "metric_log_loss": 0.6926759485334189,
    "metric_brier_score": 0.249758558899276,
    "metric_accuracy": 0.5179589131875414,
    "metric_mean_forward_return": 2.2535562542237453e-05,
    "metric_sample_count": 15090.0,
    "metric_positive_rate": 0.5046388336646785,
    "metric_signal_coverage_up_0_5": 0.4836978131212723,
    "metric_signal_coverage_down_0_5": 0.5163021868787276,
    "metric_signal_coverage_up_0_55": 0.1544731610337972,
    "metric_signal_coverage_down_0_55": 0.1540755467196819,
    "metric_signal_coverage_up_0_6": 0.009145129224652,
    "metric_signal_coverage_down_0_6": 0.0050364479787939,
    "metric_signal_coverage_up_0_65": 0.0003313452617627,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 3.2403239186256726e-05,
    "metric_mean_forward_return_down_0_5": -1.3291027588483547e-05,
    "metric_mean_forward_return_up_0_55": 6.339254316538435e-05,
    "metric_mean_forward_return_down_0_55": 0.0005789169900937,
    "metric_mean_forward_return_up_0_6": 0.0018042926517104,
    "metric_mean_forward_return_down_0_6": 0.0040642315310939,
    "metric_mean_forward_return_up_0_65": 0.0120822396294625,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 15

- asset: BTC/USDT
- timeframe: 1h
- horizon: 4h
- model: random_forest
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5524143356297964,
    "metric_pr_auc": 0.5394800053797146,
    "metric_balanced_accuracy": 0.5405404179693664,
    "metric_log_loss": 0.6901158065011955,
    "metric_brier_score": 0.2484617756854528,
    "metric_accuracy": 0.5401826484018265,
    "metric_mean_forward_return": -0.0003814488729362,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4966894977168949,
    "metric_signal_coverage_up_0_5": 0.5537671232876712,
    "metric_signal_coverage_down_0_5": 0.4462328767123287,
    "metric_signal_coverage_up_0_55": 0.2835616438356164,
    "metric_signal_coverage_down_0_55": 0.2110730593607306,
    "metric_signal_coverage_up_0_6": 0.0563926940639269,
    "metric_signal_coverage_down_0_6": 0.0490867579908675,
    "metric_signal_coverage_up_0_65": 0.0092465753424657,
    "metric_signal_coverage_down_0_65": 0.0074200913242009,
    "metric_signal_coverage_up_0_7": 0.0011415525114155,
    "metric_signal_coverage_down_0_7": 0.0007990867579908,
    "metric_mean_forward_return_up_0_5": -0.0003669223700509,
    "metric_mean_forward_return_down_0_5": 0.0003994760066012,
    "metric_mean_forward_return_up_0_55": -0.000437128299905,
    "metric_mean_forward_return_down_0_55": 0.000120057757396,
    "metric_mean_forward_return_up_0_6": 0.0001036048713235,
    "metric_mean_forward_return_down_0_6": 0.0001560278406721,
    "metric_mean_forward_return_up_0_65": -0.0008401029738199,
    "metric_mean_forward_return_down_0_65": 0.0011917156377267,
    "metric_mean_forward_return_up_0_7": 0.0045554975401693,
    "metric_mean_forward_return_down_0_7": 0.002831702229765
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5752260708303983,
    "metric_pr_auc": 0.5639129267410736,
    "metric_balanced_accuracy": 0.5613210184408375,
    "metric_log_loss": 0.6847725894049226,
    "metric_brier_score": 0.245828419298258,
    "metric_accuracy": 0.561822125813449,
    "metric_mean_forward_return": 0.0004712740487291,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5089622102979792,
    "metric_signal_coverage_up_0_5": 0.5290558282909008,
    "metric_signal_coverage_down_0_5": 0.4709441717090992,
    "metric_signal_coverage_up_0_55": 0.2431784450279712,
    "metric_signal_coverage_down_0_55": 0.208927959812764,
    "metric_signal_coverage_up_0_6": 0.0336796437949537,
    "metric_signal_coverage_down_0_6": 0.0308254366936864,
    "metric_signal_coverage_up_0_65": 0.0035392168055714,
    "metric_signal_coverage_down_0_65": 0.001826692544811,
    "metric_signal_coverage_up_0_7": 0.0002283365681013,
    "metric_signal_coverage_down_0_7": 0.0002283365681013,
    "metric_mean_forward_return_up_0_5": 0.0006537639570669,
    "metric_mean_forward_return_down_0_5": -0.000266265991702,
    "metric_mean_forward_return_up_0_55": 0.0006544178680143,
    "metric_mean_forward_return_down_0_55": -0.0004296796172304,
    "metric_mean_forward_return_up_0_6": 0.0015324052770988,
    "metric_mean_forward_return_down_0_6": -0.0004850072765512,
    "metric_mean_forward_return_up_0_65": 0.0034465927367254,
    "metric_mean_forward_return_down_0_65": 0.0002898531705552,
    "metric_mean_forward_return_up_0_7": 0.0114761150838801,
    "metric_mean_forward_return_down_0_7": 0.012520018831622
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.55891739654448,
    "metric_pr_auc": 0.5729466222077599,
    "metric_balanced_accuracy": 0.5395511010007172,
    "metric_log_loss": 0.6882540090850966,
    "metric_brier_score": 0.247563767098093,
    "metric_accuracy": 0.5393897996357013,
    "metric_mean_forward_return": 0.0004221157517958,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.519808743169399,
    "metric_signal_coverage_up_0_5": 0.497495446265938,
    "metric_signal_coverage_down_0_5": 0.5025045537340619,
    "metric_signal_coverage_up_0_55": 0.2484061930783242,
    "metric_signal_coverage_down_0_55": 0.2608151183970856,
    "metric_signal_coverage_up_0_6": 0.043943533697632,
    "metric_signal_coverage_down_0_6": 0.0660291438979963,
    "metric_signal_coverage_up_0_65": 0.0012522768670309,
    "metric_signal_coverage_down_0_65": 0.002959927140255,
    "metric_signal_coverage_up_0_7": 0.0002276867030965,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0007473105830708,
    "metric_mean_forward_return_down_0_5": -0.0001001625545435,
    "metric_mean_forward_return_up_0_55": 0.0010455967372467,
    "metric_mean_forward_return_down_0_55": 7.015890391713982e-05,
    "metric_mean_forward_return_up_0_6": 0.0021565097438348,
    "metric_mean_forward_return_down_0_6": 0.0002499397532897,
    "metric_mean_forward_return_up_0_65": 0.0026617124807157,
    "metric_mean_forward_return_down_0_65": 0.0022508834258029,
    "metric_mean_forward_return_up_0_7": 0.0013716989772248,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5331711351254016,
    "metric_pr_auc": 0.5261950914187128,
    "metric_balanced_accuracy": 0.5244069419404143,
    "metric_log_loss": 0.6943903048435789,
    "metric_brier_score": 0.2505815403768669,
    "metric_accuracy": 0.524387011265739,
    "metric_mean_forward_return": 2.2535562542237453e-05,
    "metric_sample_count": 15090.0,
    "metric_positive_rate": 0.5046388336646785,
    "metric_signal_coverage_up_0_5": 0.498078197481776,
    "metric_signal_coverage_down_0_5": 0.5019218025182239,
    "metric_signal_coverage_up_0_55": 0.2579191517561299,
    "metric_signal_coverage_down_0_55": 0.2642809807819748,
    "metric_signal_coverage_up_0_6": 0.0646123260437375,
    "metric_signal_coverage_down_0_6": 0.0577866136514247,
    "metric_signal_coverage_up_0_65": 0.0050364479787939,
    "metric_signal_coverage_down_0_65": 0.003976143141153,
    "metric_signal_coverage_up_0_7": 6.626905235255136e-05,
    "metric_signal_coverage_down_0_7": 0.0003976143141153,
    "metric_mean_forward_return_up_0_5": -2.9645524940423176e-05,
    "metric_mean_forward_return_down_0_5": -7.431705891399312e-05,
    "metric_mean_forward_return_up_0_55": -0.0001108884848773,
    "metric_mean_forward_return_down_0_55": -0.0001301850702281,
    "metric_mean_forward_return_up_0_6": -0.0005723808885908,
    "metric_mean_forward_return_down_0_6": -0.0002252177513874,
    "metric_mean_forward_return_up_0_65": -0.0023664899794719,
    "metric_mean_forward_return_down_0_65": 0.000300177297497,
    "metric_mean_forward_return_up_0_7": -0.001816961811127,
    "metric_mean_forward_return_down_0_7": 0.0029939819087742
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 16

- asset: BTC/USDT
- timeframe: 1h
- horizon: 8h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: B
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.538528302436173,
    "metric_pr_auc": 0.5145742621181124,
    "metric_balanced_accuracy": 0.5254398479728946,
    "metric_log_loss": 0.6919177102659768,
    "metric_brier_score": 0.2493585954917408,
    "metric_accuracy": 0.5247716894977169,
    "metric_mean_forward_return": -0.0007698167163356,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.483675799086758,
    "metric_signal_coverage_up_0_5": 0.5196347031963471,
    "metric_signal_coverage_down_0_5": 0.480365296803653,
    "metric_signal_coverage_up_0_55": 0.1745433789954338,
    "metric_signal_coverage_down_0_55": 0.1453196347031963,
    "metric_signal_coverage_up_0_6": 0.0380136986301369,
    "metric_signal_coverage_down_0_6": 0.021917808219178,
    "metric_signal_coverage_up_0_65": 0.0085616438356164,
    "metric_signal_coverage_down_0_65": 0.0017123287671232,
    "metric_signal_coverage_up_0_7": 0.0015981735159817,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0004030353943215,
    "metric_mean_forward_return_down_0_5": 0.0011665820627728,
    "metric_mean_forward_return_up_0_55": 0.0005480613256906,
    "metric_mean_forward_return_down_0_55": 0.0030997240173869,
    "metric_mean_forward_return_up_0_6": -0.000639414487773,
    "metric_mean_forward_return_down_0_6": 0.0055072445720759,
    "metric_mean_forward_return_up_0_65": -0.0056464551004086,
    "metric_mean_forward_return_down_0_65": 0.0110913685165942,
    "metric_mean_forward_return_up_0_7": -0.0179513158438686,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5714668428808048,
    "metric_pr_auc": 0.5788491758931247,
    "metric_balanced_accuracy": 0.5531798001918655,
    "metric_log_loss": 0.6866309469663743,
    "metric_brier_score": 0.2467601166812316,
    "metric_accuracy": 0.5521178216691404,
    "metric_mean_forward_return": 0.0009429702435061,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.512729763671652,
    "metric_signal_coverage_up_0_5": 0.4596415115880808,
    "metric_signal_coverage_down_0_5": 0.5403584884119191,
    "metric_signal_coverage_up_0_55": 0.0786619477109259,
    "metric_signal_coverage_down_0_55": 0.1003539216805571,
    "metric_signal_coverage_up_0_6": 0.0090192944400045,
    "metric_signal_coverage_down_0_6": 0.0054800776344331,
    "metric_signal_coverage_up_0_65": 0.0009133462724055,
    "metric_signal_coverage_down_0_65": 0.0006850097043041,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0013774301050153,
    "metric_mean_forward_return_down_0_5": -0.0005734085696341,
    "metric_mean_forward_return_up_0_55": 0.0027652905089818,
    "metric_mean_forward_return_down_0_55": -0.0005194650532548,
    "metric_mean_forward_return_up_0_6": 0.0071396964678946,
    "metric_mean_forward_return_down_0_6": 0.0058524508901458,
    "metric_mean_forward_return_up_0_65": 0.0139307131717091,
    "metric_mean_forward_return_down_0_65": 0.004796183392288,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5298001384513196,
    "metric_pr_auc": 0.5659737128318968,
    "metric_balanced_accuracy": 0.5195885592264354,
    "metric_log_loss": 0.6928863933087341,
    "metric_brier_score": 0.2498660205576637,
    "metric_accuracy": 0.5161657559198543,
    "metric_mean_forward_return": 0.0008427211848131,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5347222222222222,
    "metric_signal_coverage_up_0_5": 0.4520719489981785,
    "metric_signal_coverage_down_0_5": 0.5479280510018215,
    "metric_signal_coverage_up_0_55": 0.1294398907103825,
    "metric_signal_coverage_down_0_55": 0.1568761384335154,
    "metric_signal_coverage_up_0_6": 0.0198087431693989,
    "metric_signal_coverage_down_0_6": 0.016507285974499,
    "metric_signal_coverage_up_0_65": 0.0015938069216757,
    "metric_signal_coverage_down_0_65": 0.0010245901639344,
    "metric_signal_coverage_up_0_7": 0.0001138433515482,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0008561883174269,
    "metric_mean_forward_return_down_0_5": -0.0008316100309363,
    "metric_mean_forward_return_up_0_55": 0.0021664423547538,
    "metric_mean_forward_return_down_0_55": -0.0002265840492122,
    "metric_mean_forward_return_up_0_6": 0.0052106350181567,
    "metric_mean_forward_return_down_0_6": 0.0009358364846317,
    "metric_mean_forward_return_up_0_65": 0.0065236344250267,
    "metric_mean_forward_return_down_0_65": -0.0014114786079557,
    "metric_mean_forward_return_up_0_7": 0.0410629568792573,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5189274642082491,
    "metric_pr_auc": 0.5210500357831056,
    "metric_balanced_accuracy": 0.5126105261084963,
    "metric_log_loss": 0.6944036452740758,
    "metric_brier_score": 0.2506172144933971,
    "metric_accuracy": 0.5117327323346149,
    "metric_mean_forward_return": 4.242877767187773e-05,
    "metric_sample_count": 15086.0,
    "metric_positive_rate": 0.5101418533739891,
    "metric_signal_coverage_up_0_5": 0.4569799814397454,
    "metric_signal_coverage_down_0_5": 0.5430200185602545,
    "metric_signal_coverage_up_0_55": 0.1156038711388041,
    "metric_signal_coverage_down_0_55": 0.1519952273631181,
    "metric_signal_coverage_up_0_6": 0.0085509744133633,
    "metric_signal_coverage_down_0_6": 0.0072252419461752,
    "metric_signal_coverage_up_0_65": 0.0007954394803128,
    "metric_signal_coverage_down_0_65": 0.0005965796102346,
    "metric_signal_coverage_up_0_7": 6.628662335940607e-05,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001526484046759,
    "metric_mean_forward_return_down_0_5": 5.032685081521972e-05,
    "metric_mean_forward_return_up_0_55": 0.0003381528606593,
    "metric_mean_forward_return_down_0_55": 0.0007523450658892,
    "metric_mean_forward_return_up_0_6": 0.0025682738562295,
    "metric_mean_forward_return_down_0_6": 0.0029030900139647,
    "metric_mean_forward_return_up_0_65": 0.0157636382885839,
    "metric_mean_forward_return_down_0_65": 0.0384941962549557,
    "metric_mean_forward_return_up_0_7": 0.0569234578039707,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is B; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 17

- asset: ETH/USDT
- timeframe: 1h
- horizon: 12h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2022; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5434274645429159,
    "metric_pr_auc": 0.514877288353102,
    "metric_balanced_accuracy": 0.5294938472021805,
    "metric_log_loss": 0.6909980336680296,
    "metric_brier_score": 0.2488749100166593,
    "metric_accuracy": 0.5325342465753424,
    "metric_mean_forward_return": -0.0010615736882224,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4821917808219178,
    "metric_signal_coverage_up_0_5": 0.4135844748858447,
    "metric_signal_coverage_down_0_5": 0.5864155251141553,
    "metric_signal_coverage_up_0_55": 0.1248858447488584,
    "metric_signal_coverage_down_0_55": 0.1527397260273972,
    "metric_signal_coverage_up_0_6": 0.035958904109589,
    "metric_signal_coverage_down_0_6": 0.0133561643835616,
    "metric_signal_coverage_up_0_65": 0.0116438356164383,
    "metric_signal_coverage_down_0_65": 0.0013698630136986,
    "metric_signal_coverage_up_0_7": 0.0047945205479452,
    "metric_signal_coverage_down_0_7": 0.0001141552511415,
    "metric_mean_forward_return_up_0_5": 0.0013786519487595,
    "metric_mean_forward_return_down_0_5": 0.0027826049287881,
    "metric_mean_forward_return_up_0_55": 0.0032919381291297,
    "metric_mean_forward_return_down_0_55": 0.0040395735059658,
    "metric_mean_forward_return_up_0_6": 0.0040630743715528,
    "metric_mean_forward_return_down_0_6": 0.0085547059496685,
    "metric_mean_forward_return_up_0_65": -0.0038479579314187,
    "metric_mean_forward_return_down_0_65": 0.0070493097601124,
    "metric_mean_forward_return_up_0_7": -0.0276399302888991,
    "metric_mean_forward_return_down_0_7": 0.0047137987358851
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5381370042799041,
    "metric_pr_auc": 0.5228112179004665,
    "metric_balanced_accuracy": 0.5169447938575966,
    "metric_log_loss": 0.6911551733000175,
    "metric_brier_score": 0.2490080805325101,
    "metric_accuracy": 0.5207215435552004,
    "metric_mean_forward_return": 0.0010322443448566,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.4933211553830346,
    "metric_signal_coverage_up_0_5": 0.217033907980363,
    "metric_signal_coverage_down_0_5": 0.782966092019637,
    "metric_signal_coverage_up_0_55": 0.0260303687635574,
    "metric_signal_coverage_down_0_55": 0.1177075008562621,
    "metric_signal_coverage_up_0_6": 0.0013700194086082,
    "metric_signal_coverage_down_0_6": 0.0023975339650645,
    "metric_signal_coverage_up_0_65": 0.0,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0028760797179782,
    "metric_mean_forward_return_down_0_5": -0.0005211432885276,
    "metric_mean_forward_return_up_0_55": 0.0046780011126542,
    "metric_mean_forward_return_down_0_55": 0.0001972333875805,
    "metric_mean_forward_return_up_0_6": 0.006353925810016,
    "metric_mean_forward_return_down_0_6": 0.0027453802369857,
    "metric_mean_forward_return_up_0_65": NaN,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5370435575964128,
    "metric_pr_auc": 0.5642394603371553,
    "metric_balanced_accuracy": 0.5299899230080405,
    "metric_log_loss": 0.6917858386153918,
    "metric_brier_score": 0.2493357887049302,
    "metric_accuracy": 0.5229963570127505,
    "metric_mean_forward_return": 0.0007948724836649,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5235655737704918,
    "metric_signal_coverage_up_0_5": 0.353028233151184,
    "metric_signal_coverage_down_0_5": 0.6469717668488161,
    "metric_signal_coverage_up_0_55": 0.0584016393442623,
    "metric_signal_coverage_down_0_55": 0.1214708561020036,
    "metric_signal_coverage_up_0_6": 0.0107012750455373,
    "metric_signal_coverage_down_0_6": 0.0069444444444444,
    "metric_signal_coverage_up_0_65": 0.0028460837887067,
    "metric_signal_coverage_down_0_65": 0.0007969034608378,
    "metric_signal_coverage_up_0_7": 0.0014799635701275,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0016324715853113,
    "metric_mean_forward_return_down_0_5": -0.0003378260620204,
    "metric_mean_forward_return_up_0_55": 0.0048755490707628,
    "metric_mean_forward_return_down_0_55": -4.512777577804855e-05,
    "metric_mean_forward_return_up_0_6": 0.0057382496428631,
    "metric_mean_forward_return_down_0_6": -0.001904929135113,
    "metric_mean_forward_return_up_0_65": 0.0103541117467995,
    "metric_mean_forward_return_down_0_65": -0.0028345812255459,
    "metric_mean_forward_return_up_0_7": 0.0162002175316158,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5039234696185537,
    "metric_pr_auc": 0.5163901499087452,
    "metric_balanced_accuracy": 0.4987665296016877,
    "metric_log_loss": 0.6965208981381377,
    "metric_brier_score": 0.2516617425889311,
    "metric_accuracy": 0.4955247629781873,
    "metric_mean_forward_return": 0.000166204106655,
    "metric_sample_count": 15083.0,
    "metric_positive_rate": 0.5164755022210435,
    "metric_signal_coverage_up_0_5": 0.4015779354239873,
    "metric_signal_coverage_down_0_5": 0.5984220645760128,
    "metric_signal_coverage_up_0_55": 0.0894384406285221,
    "metric_signal_coverage_down_0_55": 0.1211297487237287,
    "metric_signal_coverage_up_0_6": 0.0157793542398727,
    "metric_signal_coverage_down_0_6": 0.0079559769276669,
    "metric_signal_coverage_up_0_65": 0.0032486905787973,
    "metric_signal_coverage_down_0_65": 0.0013259961546111,
    "metric_signal_coverage_up_0_7": 0.0011270967314194,
    "metric_signal_coverage_down_0_7": 0.0001988994231916,
    "metric_mean_forward_return_up_0_5": -0.0001777380270253,
    "metric_mean_forward_return_down_0_5": -0.0003970103889175,
    "metric_mean_forward_return_up_0_55": -0.0010172833084886,
    "metric_mean_forward_return_down_0_55": 0.0011966684413454,
    "metric_mean_forward_return_up_0_6": 0.0046960193347014,
    "metric_mean_forward_return_down_0_6": 0.0070917095587148,
    "metric_mean_forward_return_up_0_65": 0.0190142652202296,
    "metric_mean_forward_return_down_0_65": 0.0217734527645166,
    "metric_mean_forward_return_up_0_7": 0.007886856820216,
    "metric_mean_forward_return_down_0_7": 0.0419113319255998
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 18

- asset: ETH/USDT
- timeframe: 1h
- horizon: 1h
- model: hist_gradient_boosting
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5602244171254588,
    "metric_pr_auc": 0.5404874141970268,
    "metric_balanced_accuracy": 0.5393362112112112,
    "metric_log_loss": 0.6884189906661146,
    "metric_brier_score": 0.2476411810686549,
    "metric_accuracy": 0.538013698630137,
    "metric_mean_forward_return": -8.630355504433286e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4931506849315068,
    "metric_signal_coverage_up_0_5": 0.5960045662100457,
    "metric_signal_coverage_down_0_5": 0.4039954337899543,
    "metric_signal_coverage_up_0_55": 0.1899543378995433,
    "metric_signal_coverage_down_0_55": 0.1260273972602739,
    "metric_signal_coverage_up_0_6": 0.0073059360730593,
    "metric_signal_coverage_down_0_6": 0.0121004566210045,
    "metric_signal_coverage_up_0_65": 0.0,
    "metric_signal_coverage_down_0_65": 0.0004566210045662,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -6.408535237441562e-05,
    "metric_mean_forward_return_down_0_5": 0.0001190815251318,
    "metric_mean_forward_return_up_0_55": -3.793301729540857e-06,
    "metric_mean_forward_return_down_0_55": 0.0002380671107698,
    "metric_mean_forward_return_up_0_6": -0.0003292497753812,
    "metric_mean_forward_return_down_0_6": -0.0003469716075219,
    "metric_mean_forward_return_up_0_65": NaN,
    "metric_mean_forward_return_down_0_65": 0.0020969636102934,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5625137723270736,
    "metric_pr_auc": 0.5611795426066736,
    "metric_balanced_accuracy": 0.5417288604882169,
    "metric_log_loss": 0.687450905251339,
    "metric_brier_score": 0.2471616940695258,
    "metric_accuracy": 0.5417285078205275,
    "metric_mean_forward_return": 8.594962723625279e-05,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5051946569243064,
    "metric_signal_coverage_up_0_5": 0.5003995889941775,
    "metric_signal_coverage_down_0_5": 0.4996004110058226,
    "metric_signal_coverage_up_0_55": 0.1209042128096814,
    "metric_signal_coverage_down_0_55": 0.1703390798036305,
    "metric_signal_coverage_up_0_6": 0.0068500970430414,
    "metric_signal_coverage_down_0_6": 0.0328804658065989,
    "metric_signal_coverage_up_0_65": 0.0,
    "metric_signal_coverage_down_0_65": 0.0012558511245575,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 5.433995545145254e-05,
    "metric_mean_forward_return_down_0_5": -0.0001176098629384,
    "metric_mean_forward_return_up_0_55": 0.0003652402973001,
    "metric_mean_forward_return_down_0_55": -0.0001238603017757,
    "metric_mean_forward_return_up_0_6": 0.0002718030905452,
    "metric_mean_forward_return_down_0_6": -6.135247073152383e-05,
    "metric_mean_forward_return_up_0_65": NaN,
    "metric_mean_forward_return_down_0_65": -0.000259951944255,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5603677219906796,
    "metric_pr_auc": 0.558698320021798,
    "metric_balanced_accuracy": 0.5450290673374476,
    "metric_log_loss": 0.6877764232284816,
    "metric_brier_score": 0.2473209936479567,
    "metric_accuracy": 0.5456511839708561,
    "metric_mean_forward_return": 6.670945482370432e-05,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5084244080145719,
    "metric_signal_coverage_up_0_5": 0.5376821493624773,
    "metric_signal_coverage_down_0_5": 0.4623178506375228,
    "metric_signal_coverage_up_0_55": 0.1982012750455373,
    "metric_signal_coverage_down_0_55": 0.1966074681238615,
    "metric_signal_coverage_up_0_6": 0.01775956284153,
    "metric_signal_coverage_down_0_6": 0.026639344262295,
    "metric_signal_coverage_up_0_65": 0.0007969034608378,
    "metric_signal_coverage_down_0_65": 0.002959927140255,
    "metric_signal_coverage_up_0_7": 0.0001138433515482,
    "metric_signal_coverage_down_0_7": 0.0005692167577413,
    "metric_mean_forward_return_up_0_5": 9.041987418578689e-05,
    "metric_mean_forward_return_down_0_5": -3.913390430730048e-05,
    "metric_mean_forward_return_up_0_55": 0.0002778203150535,
    "metric_mean_forward_return_down_0_55": 2.341089920232674e-05,
    "metric_mean_forward_return_up_0_6": 0.0006590789916083,
    "metric_mean_forward_return_down_0_6": -0.0002338050405124,
    "metric_mean_forward_return_up_0_65": 0.006474824069364,
    "metric_mean_forward_return_down_0_65": 4.701098000118821e-05,
    "metric_mean_forward_return_up_0_7": 0.0166156550290963,
    "metric_mean_forward_return_down_0_7": -0.0004252311372462
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5333716555624386,
    "metric_pr_auc": 0.5304629522578381,
    "metric_balanced_accuracy": 0.5238652581645223,
    "metric_log_loss": 0.6923406606762729,
    "metric_brier_score": 0.2495884487137814,
    "metric_accuracy": 0.5243805485623426,
    "metric_mean_forward_return": 1.3060296590494272e-05,
    "metric_sample_count": 15094.0,
    "metric_positive_rate": 0.5051013647807076,
    "metric_signal_coverage_up_0_5": 0.5507486418444415,
    "metric_signal_coverage_down_0_5": 0.4492513581555585,
    "metric_signal_coverage_up_0_55": 0.202994567377766,
    "metric_signal_coverage_down_0_55": 0.1731813965814231,
    "metric_signal_coverage_up_0_6": 0.0152378428514641,
    "metric_signal_coverage_down_0_6": 0.0160991122300251,
    "metric_signal_coverage_up_0_65": 0.000132502981317,
    "metric_signal_coverage_down_0_65": 0.0002650059626341,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -5.7548799564794045e-05,
    "metric_mean_forward_return_down_0_5": -9.962177960758788e-05,
    "metric_mean_forward_return_up_0_55": -0.0002883417447671,
    "metric_mean_forward_return_down_0_55": -4.44860191075987e-05,
    "metric_mean_forward_return_up_0_6": -0.0006849376943268,
    "metric_mean_forward_return_down_0_6": -0.0003034514614374,
    "metric_mean_forward_return_up_0_65": -0.0073132635797989,
    "metric_mean_forward_return_down_0_65": 0.0004938916158072,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 19

- asset: ETH/USDT
- timeframe: 1h
- horizon: 1h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2024; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5474535472972972,
    "metric_pr_auc": 0.5259197807375651,
    "metric_balanced_accuracy": 0.536082957957958,
    "metric_log_loss": 0.6906799755820494,
    "metric_brier_score": 0.2487552900688466,
    "metric_accuracy": 0.5365296803652968,
    "metric_mean_forward_return": -8.630355504433286e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4931506849315068,
    "metric_signal_coverage_up_0_5": 0.4668949771689498,
    "metric_signal_coverage_down_0_5": 0.5331050228310502,
    "metric_signal_coverage_up_0_55": 0.1226027397260274,
    "metric_signal_coverage_down_0_55": 0.1263698630136986,
    "metric_signal_coverage_up_0_6": 0.0212328767123287,
    "metric_signal_coverage_down_0_6": 0.0070776255707762,
    "metric_signal_coverage_up_0_65": 0.0025114155251141,
    "metric_signal_coverage_down_0_65": 0.0003424657534246,
    "metric_signal_coverage_up_0_7": 0.0003424657534246,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0001180408792813,
    "metric_mean_forward_return_down_0_5": 5.850791133350039e-05,
    "metric_mean_forward_return_up_0_55": -0.0006106077090784,
    "metric_mean_forward_return_down_0_55": -0.0002118495292419,
    "metric_mean_forward_return_up_0_6": -0.0021540088579545,
    "metric_mean_forward_return_down_0_6": 0.0013933446412166,
    "metric_mean_forward_return_up_0_65": 0.0012484247139503,
    "metric_mean_forward_return_down_0_65": 0.0090020750512656,
    "metric_mean_forward_return_up_0_7": 0.0206372711980551,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5524829296144792,
    "metric_pr_auc": 0.5520552295510246,
    "metric_balanced_accuracy": 0.5387752601294716,
    "metric_log_loss": 0.6893315532370429,
    "metric_brier_score": 0.2480999437229364,
    "metric_accuracy": 0.5377326178787533,
    "metric_mean_forward_return": 8.594962723625279e-05,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5051946569243064,
    "metric_signal_coverage_up_0_5": 0.4000456673136203,
    "metric_signal_coverage_down_0_5": 0.5999543326863798,
    "metric_signal_coverage_up_0_55": 0.0529740837995205,
    "metric_signal_coverage_down_0_55": 0.1117707500856262,
    "metric_signal_coverage_up_0_6": 0.0030825436693686,
    "metric_signal_coverage_down_0_6": 0.0009133462724055,
    "metric_signal_coverage_up_0_65": 0.0005708414202534,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0001141682840506,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 6.008801207457352e-05,
    "metric_mean_forward_return_down_0_5": -0.0001031939849006,
    "metric_mean_forward_return_up_0_55": 0.0004190496931851,
    "metric_mean_forward_return_down_0_55": -0.0002192762544066,
    "metric_mean_forward_return_up_0_6": 0.0033826239858167,
    "metric_mean_forward_return_down_0_6": -0.0023611822140178,
    "metric_mean_forward_return_up_0_65": 0.0119156230119521,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0203200017498388,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5580920492996646,
    "metric_pr_auc": 0.5532844741996858,
    "metric_balanced_accuracy": 0.5443587253972011,
    "metric_log_loss": 0.6886538685574898,
    "metric_brier_score": 0.2477587927387335,
    "metric_accuracy": 0.5432604735883424,
    "metric_mean_forward_return": 6.670945482370432e-05,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5084244080145719,
    "metric_signal_coverage_up_0_5": 0.4355646630236794,
    "metric_signal_coverage_down_0_5": 0.5644353369763205,
    "metric_signal_coverage_up_0_55": 0.100523679417122,
    "metric_signal_coverage_down_0_55": 0.1286429872495446,
    "metric_signal_coverage_up_0_6": 0.0111566484517304,
    "metric_signal_coverage_down_0_6": 0.0033014571948998,
    "metric_signal_coverage_up_0_65": 0.0005692167577413,
    "metric_signal_coverage_down_0_65": 0.0001138433515482,
    "metric_signal_coverage_up_0_7": 0.0002276867030965,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001325298451696,
    "metric_mean_forward_return_down_0_5": -1.59170358113005e-05,
    "metric_mean_forward_return_up_0_55": 0.0004420182743346,
    "metric_mean_forward_return_down_0_55": 0.0001574042744016,
    "metric_mean_forward_return_up_0_6": -0.0002256427314249,
    "metric_mean_forward_return_down_0_6": -0.0009488579435079,
    "metric_mean_forward_return_up_0_65": -0.0071980471379005,
    "metric_mean_forward_return_down_0_65": 0.0093579954361944,
    "metric_mean_forward_return_up_0_7": 0.010019182668651,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5296697984663383,
    "metric_pr_auc": 0.5273099734913544,
    "metric_balanced_accuracy": 0.5236907054591222,
    "metric_log_loss": 0.6927143044557373,
    "metric_brier_score": 0.2497741914325043,
    "metric_accuracy": 0.523320524711806,
    "metric_mean_forward_return": 1.3060296590494272e-05,
    "metric_sample_count": 15094.0,
    "metric_positive_rate": 0.5051013647807076,
    "metric_signal_coverage_up_0_5": 0.4639591890817543,
    "metric_signal_coverage_down_0_5": 0.5360408109182456,
    "metric_signal_coverage_up_0_55": 0.1156751026898105,
    "metric_signal_coverage_down_0_55": 0.1353517954153968,
    "metric_signal_coverage_up_0_6": 0.0132502981317079,
    "metric_signal_coverage_down_0_6": 0.0045051013647807,
    "metric_signal_coverage_up_0_65": 0.0012587783225122,
    "metric_signal_coverage_down_0_65": 0.0001987544719756,
    "metric_signal_coverage_up_0_7": 0.0001987544719756,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 2.200043563797966e-06,
    "metric_mean_forward_return_down_0_5": -2.246016705718e-05,
    "metric_mean_forward_return_up_0_55": -0.0002638278128506,
    "metric_mean_forward_return_down_0_55": -0.0003057198788115,
    "metric_mean_forward_return_up_0_6": -0.0005009968545209,
    "metric_mean_forward_return_down_0_6": 0.000609524328949,
    "metric_mean_forward_return_up_0_65": 0.0022404132783307,
    "metric_mean_forward_return_down_0_65": 0.0021067245762746,
    "metric_mean_forward_return_up_0_7": 0.0075327038260239,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 20

- asset: ETH/USDT
- timeframe: 1h
- horizon: 1h
- model: random_forest
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5551348744577911,
    "metric_pr_auc": 0.5377759124861423,
    "metric_balanced_accuracy": 0.5433996496496496,
    "metric_log_loss": 0.688805314405448,
    "metric_brier_score": 0.2478309533041245,
    "metric_accuracy": 0.5429223744292238,
    "metric_mean_forward_return": -8.630355504433286e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4931506849315068,
    "metric_signal_coverage_up_0_5": 0.5342465753424658,
    "metric_signal_coverage_down_0_5": 0.4657534246575342,
    "metric_signal_coverage_up_0_55": 0.2015981735159817,
    "metric_signal_coverage_down_0_55": 0.2063926940639269,
    "metric_signal_coverage_up_0_6": 0.027283105022831,
    "metric_signal_coverage_down_0_6": 0.040296803652968,
    "metric_signal_coverage_up_0_65": 0.0027397260273972,
    "metric_signal_coverage_down_0_65": 0.0018264840182648,
    "metric_signal_coverage_up_0_7": 0.0001141552511415,
    "metric_signal_coverage_down_0_7": 0.0009132420091324,
    "metric_mean_forward_return_up_0_5": 9.634838141654568e-06,
    "metric_mean_forward_return_down_0_5": 0.0001963505354635,
    "metric_mean_forward_return_up_0_55": 0.0001239362572556,
    "metric_mean_forward_return_down_0_55": 0.0001487736815906,
    "metric_mean_forward_return_up_0_6": 0.0010206806363547,
    "metric_mean_forward_return_down_0_6": 0.0004315728420377,
    "metric_mean_forward_return_up_0_65": 0.004629603983497,
    "metric_mean_forward_return_down_0_65": -0.0018098716673866,
    "metric_mean_forward_return_up_0_7": 0.025378439511819,
    "metric_mean_forward_return_down_0_7": -0.0014518752170885
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5619885337066787,
    "metric_pr_auc": 0.5561533177954284,
    "metric_balanced_accuracy": 0.5449205728453771,
    "metric_log_loss": 0.6875212493318611,
    "metric_brier_score": 0.2471979903317647,
    "metric_accuracy": 0.5446968832058454,
    "metric_mean_forward_return": 8.594962723625279e-05,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5051946569243064,
    "metric_signal_coverage_up_0_5": 0.4789359515926475,
    "metric_signal_coverage_down_0_5": 0.5210640484073524,
    "metric_signal_coverage_up_0_55": 0.1492179472542527,
    "metric_signal_coverage_down_0_55": 0.2283365681013814,
    "metric_signal_coverage_up_0_6": 0.0084484530197511,
    "metric_signal_coverage_down_0_6": 0.0358488411919168,
    "metric_signal_coverage_up_0_65": 0.0006850097043041,
    "metric_signal_coverage_down_0_65": 0.0007991779883548,
    "metric_signal_coverage_up_0_7": 0.0001141682840506,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 9.362680049970438e-05,
    "metric_mean_forward_return_down_0_5": -7.889315444042031e-05,
    "metric_mean_forward_return_up_0_55": -7.873481368847901e-06,
    "metric_mean_forward_return_down_0_55": -0.0001498439993208,
    "metric_mean_forward_return_up_0_6": 0.0015464942028933,
    "metric_mean_forward_return_down_0_6": 1.8582825594362336e-06,
    "metric_mean_forward_return_up_0_65": 0.0074775567285815,
    "metric_mean_forward_return_down_0_65": 0.0005075961572835,
    "metric_mean_forward_return_up_0_7": 0.008876493784955,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.558876370630695,
    "metric_pr_auc": 0.5583992040681468,
    "metric_balanced_accuracy": 0.5406286746426658,
    "metric_log_loss": 0.6881133067463551,
    "metric_brier_score": 0.2474951683210553,
    "metric_accuracy": 0.5406420765027322,
    "metric_mean_forward_return": 6.670945482370432e-05,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5084244080145719,
    "metric_signal_coverage_up_0_5": 0.5014799635701275,
    "metric_signal_coverage_down_0_5": 0.4985200364298724,
    "metric_signal_coverage_up_0_55": 0.2043488160291439,
    "metric_signal_coverage_down_0_55": 0.248747723132969,
    "metric_signal_coverage_up_0_6": 0.0176457194899817,
    "metric_signal_coverage_down_0_6": 0.0417805100182149,
    "metric_signal_coverage_up_0_65": 0.0010245901639344,
    "metric_signal_coverage_down_0_65": 0.0018214936247723,
    "metric_signal_coverage_up_0_7": 0.0003415300546448,
    "metric_signal_coverage_down_0_7": 0.0002276867030965,
    "metric_mean_forward_return_up_0_5": 9.668945904227374e-05,
    "metric_mean_forward_return_down_0_5": -3.655144646955993e-05,
    "metric_mean_forward_return_up_0_55": 0.0003804457875946,
    "metric_mean_forward_return_down_0_55": -4.547389842078429e-05,
    "metric_mean_forward_return_up_0_6": 0.000824967098614,
    "metric_mean_forward_return_down_0_6": 9.7268087907421e-05,
    "metric_mean_forward_return_up_0_65": 0.0105237311520619,
    "metric_mean_forward_return_down_0_65": 0.0001989041871822,
    "metric_mean_forward_return_up_0_7": 0.0112168227043921,
    "metric_mean_forward_return_down_0_7": 0.0039323560974872
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5346334972629238,
    "metric_pr_auc": 0.532435981483022,
    "metric_balanced_accuracy": 0.5279630765103085,
    "metric_log_loss": 0.693089186280142,
    "metric_brier_score": 0.2499449372280842,
    "metric_accuracy": 0.5280243805485624,
    "metric_mean_forward_return": 1.3060296590494272e-05,
    "metric_sample_count": 15094.0,
    "metric_positive_rate": 0.5051013647807076,
    "metric_signal_coverage_up_0_5": 0.5062938916125613,
    "metric_signal_coverage_down_0_5": 0.4937061083874387,
    "metric_signal_coverage_up_0_55": 0.2228037630846694,
    "metric_signal_coverage_down_0_55": 0.2457930303431827,
    "metric_signal_coverage_up_0_6": 0.0271631111700013,
    "metric_signal_coverage_down_0_6": 0.0345170266330992,
    "metric_signal_coverage_up_0_65": 0.0019212932290976,
    "metric_signal_coverage_down_0_65": 0.0046376043460977,
    "metric_signal_coverage_up_0_7": 0.0003312574532926,
    "metric_signal_coverage_down_0_7": 0.0007287663972439,
    "metric_mean_forward_return_up_0_5": -3.8909176951531095e-05,
    "metric_mean_forward_return_down_0_5": -6.635481038654336e-05,
    "metric_mean_forward_return_up_0_55": -0.000255017475305,
    "metric_mean_forward_return_down_0_55": -0.0001139265371348,
    "metric_mean_forward_return_up_0_6": 0.0002451393286897,
    "metric_mean_forward_return_down_0_6": -0.0003377833569275,
    "metric_mean_forward_return_up_0_65": 0.0041939653918357,
    "metric_mean_forward_return_down_0_65": -0.0014752181727038,
    "metric_mean_forward_return_up_0_7": 0.0080007008082774,
    "metric_mean_forward_return_down_0_7": -8.340824174978379e-05
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 21

- asset: ETH/USDT
- timeframe: 1h
- horizon: 1h
- model: logistic_regression
- feature_set: crypto_core_v1+crypto_context_proxy_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2024; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5377272063730396,
    "metric_pr_auc": 0.5222534717014611,
    "metric_balanced_accuracy": 0.5241741741741741,
    "metric_log_loss": 0.6924308464103116,
    "metric_brier_score": 0.2496287053939445,
    "metric_accuracy": 0.5271689497716895,
    "metric_mean_forward_return": -8.630355504433286e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4931506849315068,
    "metric_signal_coverage_up_0_5": 0.2810502283105023,
    "metric_signal_coverage_down_0_5": 0.7189497716894977,
    "metric_signal_coverage_up_0_55": 0.0673515981735159,
    "metric_signal_coverage_down_0_55": 0.3312785388127854,
    "metric_signal_coverage_up_0_6": 0.008904109589041,
    "metric_signal_coverage_down_0_6": 0.0482876712328767,
    "metric_signal_coverage_up_0_65": 0.0009132420091324,
    "metric_signal_coverage_down_0_65": 0.0007990867579908,
    "metric_signal_coverage_up_0_7": 0.0001141552511415,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0002163152665675,
    "metric_mean_forward_return_down_0_5": 3.547966908528101e-05,
    "metric_mean_forward_return_up_0_55": -0.0004947166203592,
    "metric_mean_forward_return_down_0_55": -2.491329546661581e-05,
    "metric_mean_forward_return_up_0_6": -1.5974233630255963e-05,
    "metric_mean_forward_return_down_0_6": -0.000225100876799,
    "metric_mean_forward_return_up_0_65": -0.006810032838993,
    "metric_mean_forward_return_down_0_65": 0.0014207144029511,
    "metric_mean_forward_return_up_0_7": 0.0154827123317489,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5503267554665644,
    "metric_pr_auc": 0.5482495993274239,
    "metric_balanced_accuracy": 0.5409648841508086,
    "metric_log_loss": 0.6893267112039844,
    "metric_brier_score": 0.2480968734449435,
    "metric_accuracy": 0.5408151615481219,
    "metric_mean_forward_return": 8.594962723625279e-05,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5051946569243064,
    "metric_signal_coverage_up_0_5": 0.4860143852037903,
    "metric_signal_coverage_down_0_5": 0.5139856147962096,
    "metric_signal_coverage_up_0_55": 0.0944171709099212,
    "metric_signal_coverage_down_0_55": 0.0799177988354835,
    "metric_signal_coverage_up_0_6": 0.0034250485215207,
    "metric_signal_coverage_down_0_6": 0.0009133462724055,
    "metric_signal_coverage_up_0_65": 0.0002283365681013,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 6.016591271570644e-05,
    "metric_mean_forward_return_down_0_5": -0.000110330185369,
    "metric_mean_forward_return_up_0_55": 6.537349350664516e-05,
    "metric_mean_forward_return_down_0_55": -0.0002802527163708,
    "metric_mean_forward_return_up_0_6": 0.0032539409315777,
    "metric_mean_forward_return_down_0_6": 0.0013273593524421,
    "metric_mean_forward_return_up_0_65": 0.0150732409889309,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5584871916826365,
    "metric_pr_auc": 0.5519957433105449,
    "metric_balanced_accuracy": 0.5431648975834502,
    "metric_log_loss": 0.6886343246223748,
    "metric_brier_score": 0.2477499429062407,
    "metric_accuracy": 0.5418943533697632,
    "metric_mean_forward_return": 6.670945482370432e-05,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5084244080145719,
    "metric_signal_coverage_up_0_5": 0.4253187613843351,
    "metric_signal_coverage_down_0_5": 0.5746812386156649,
    "metric_signal_coverage_up_0_55": 0.1055327868852459,
    "metric_signal_coverage_down_0_55": 0.167235883424408,
    "metric_signal_coverage_up_0_6": 0.0119535519125683,
    "metric_signal_coverage_down_0_6": 0.0048952641165755,
    "metric_signal_coverage_up_0_65": 0.0006830601092896,
    "metric_signal_coverage_down_0_65": 0.0001138433515482,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001354939611835,
    "metric_mean_forward_return_down_0_5": -1.5802379593866426e-05,
    "metric_mean_forward_return_up_0_55": 0.0004134323084503,
    "metric_mean_forward_return_down_0_55": 3.9708582045254814e-05,
    "metric_mean_forward_return_up_0_6": -0.0014517530578759,
    "metric_mean_forward_return_down_0_6": 0.0018220563720322,
    "metric_mean_forward_return_up_0_65": 0.0100257441309741,
    "metric_mean_forward_return_down_0_65": 0.0055605076616155,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5281393499847589,
    "metric_pr_auc": 0.5259013537476654,
    "metric_balanced_accuracy": 0.5166467022339094,
    "metric_log_loss": 0.6942813128911853,
    "metric_brier_score": 0.2505363555509096,
    "metric_accuracy": 0.5188816748376839,
    "metric_mean_forward_return": 1.3060296590494272e-05,
    "metric_sample_count": 15094.0,
    "metric_positive_rate": 0.5051013647807076,
    "metric_signal_coverage_up_0_5": 0.7192261825891083,
    "metric_signal_coverage_down_0_5": 0.2807738174108917,
    "metric_signal_coverage_up_0_55": 0.3398701470783092,
    "metric_signal_coverage_down_0_55": 0.0394196369418311,
    "metric_signal_coverage_up_0_6": 0.0761229627666622,
    "metric_signal_coverage_down_0_6": 0.0019212932290976,
    "metric_signal_coverage_up_0_65": 0.0052338677620246,
    "metric_signal_coverage_down_0_65": 6.625149065853981e-05,
    "metric_signal_coverage_up_0_7": 0.0003312574532926,
    "metric_signal_coverage_down_0_7": 6.625149065853981e-05,
    "metric_mean_forward_return_up_0_5": -1.0487658125502856e-05,
    "metric_mean_forward_return_down_0_5": -7.338039956285502e-05,
    "metric_mean_forward_return_up_0_55": -6.101411247227689e-05,
    "metric_mean_forward_return_down_0_55": 0.0001980987724293,
    "metric_mean_forward_return_up_0_6": -0.0002594079029549,
    "metric_mean_forward_return_down_0_6": -0.0009088157659105,
    "metric_mean_forward_return_up_0_65": 0.0008006237683603,
    "metric_mean_forward_return_down_0_65": 0.0188952656967985,
    "metric_mean_forward_return_up_0_7": 0.0077140254364926,
    "metric_mean_forward_return_down_0_7": 0.0188952656967985
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 22

- asset: ETH/USDT
- timeframe: 1h
- horizon: 1h
- model: random_forest
- feature_set: crypto_core_v1+crypto_context_proxy_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.542149545378712,
    "metric_pr_auc": 0.52722626847673,
    "metric_balanced_accuracy": 0.5342655155155155,
    "metric_log_loss": 0.6925677598673978,
    "metric_brier_score": 0.2496893507085877,
    "metric_accuracy": 0.5323059360730593,
    "metric_mean_forward_return": -8.630355504433286e-05,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4931506849315068,
    "metric_signal_coverage_up_0_5": 0.6425799086757991,
    "metric_signal_coverage_down_0_5": 0.3574200913242009,
    "metric_signal_coverage_up_0_55": 0.3015981735159817,
    "metric_signal_coverage_down_0_55": 0.125,
    "metric_signal_coverage_up_0_6": 0.0554794520547945,
    "metric_signal_coverage_down_0_6": 0.0126712328767123,
    "metric_signal_coverage_up_0_65": 0.0074200913242009,
    "metric_signal_coverage_down_0_65": 0.0002283105022831,
    "metric_signal_coverage_up_0_7": 0.0003424657534246,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0001394284499887,
    "metric_mean_forward_return_down_0_5": -9.205877610364832e-06,
    "metric_mean_forward_return_up_0_55": -0.0001582851129183,
    "metric_mean_forward_return_down_0_55": -0.0002969145026485,
    "metric_mean_forward_return_up_0_6": 0.0001189557425491,
    "metric_mean_forward_return_down_0_6": -0.0002088427251723,
    "metric_mean_forward_return_up_0_65": 0.0014115309227457,
    "metric_mean_forward_return_down_0_65": 0.0008539466259399,
    "metric_mean_forward_return_up_0_7": 0.0051914212107074,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5572067400321723,
    "metric_pr_auc": 0.5510070740539954,
    "metric_balanced_accuracy": 0.5426446778722439,
    "metric_log_loss": 0.6881463281680088,
    "metric_brier_score": 0.2475091900063545,
    "metric_accuracy": 0.542641854092933,
    "metric_mean_forward_return": 8.594962723625279e-05,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5051946569243064,
    "metric_signal_coverage_up_0_5": 0.5001712524260761,
    "metric_signal_coverage_down_0_5": 0.499828747573924,
    "metric_signal_coverage_up_0_55": 0.1605206073752711,
    "metric_signal_coverage_down_0_55": 0.1880351638314876,
    "metric_signal_coverage_up_0_6": 0.0090192944400045,
    "metric_signal_coverage_down_0_6": 0.0189519351524146,
    "metric_signal_coverage_up_0_65": 0.0009133462724055,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0002283365681013,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 6.72609996839775e-05,
    "metric_mean_forward_return_down_0_5": -0.0001046510610659,
    "metric_mean_forward_return_up_0_55": 0.0001228881736407,
    "metric_mean_forward_return_down_0_55": -0.000164999306372,
    "metric_mean_forward_return_up_0_6": 0.0012576118637422,
    "metric_mean_forward_return_down_0_6": -0.0001876250905533,
    "metric_mean_forward_return_up_0_65": -0.0001133053090055,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0024292614590692,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5540923475751223,
    "metric_pr_auc": 0.5501010814603036,
    "metric_balanced_accuracy": 0.538057967491294,
    "metric_log_loss": 0.6894480596775772,
    "metric_brier_score": 0.2481506809769441,
    "metric_accuracy": 0.5374544626593807,
    "metric_mean_forward_return": 6.670945482370432e-05,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5084244080145719,
    "metric_signal_coverage_up_0_5": 0.4648224043715847,
    "metric_signal_coverage_down_0_5": 0.5351775956284153,
    "metric_signal_coverage_up_0_55": 0.0915300546448087,
    "metric_signal_coverage_down_0_55": 0.2370218579234972,
    "metric_signal_coverage_up_0_6": 0.0048952641165755,
    "metric_signal_coverage_down_0_6": 0.0586293260473588,
    "metric_signal_coverage_up_0_65": 0.0003415300546448,
    "metric_signal_coverage_down_0_65": 0.0007969034608378,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001528126282748,
    "metric_mean_forward_return_down_0_5": 8.074475659413957e-06,
    "metric_mean_forward_return_up_0_55": 0.0003723889974326,
    "metric_mean_forward_return_down_0_55": -4.9918592790534886e-05,
    "metric_mean_forward_return_up_0_6": 0.0017209983204254,
    "metric_mean_forward_return_down_0_6": 0.0001351843363456,
    "metric_mean_forward_return_up_0_65": 0.0184360290258036,
    "metric_mean_forward_return_down_0_65": -0.0034962162625981,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5354125666710212,
    "metric_pr_auc": 0.5355322435150851,
    "metric_balanced_accuracy": 0.5258671095715496,
    "metric_log_loss": 0.6928572823279082,
    "metric_brier_score": 0.2498362795792954,
    "metric_accuracy": 0.5256393268848549,
    "metric_mean_forward_return": 1.3060296590494272e-05,
    "metric_sample_count": 15094.0,
    "metric_positive_rate": 0.5051013647807076,
    "metric_signal_coverage_up_0_5": 0.4779382536107062,
    "metric_signal_coverage_down_0_5": 0.5220617463892938,
    "metric_signal_coverage_up_0_55": 0.2160461110374983,
    "metric_signal_coverage_down_0_55": 0.2302239300384258,
    "metric_signal_coverage_up_0_6": 0.0402146548297336,
    "metric_signal_coverage_down_0_6": 0.0361733138995627,
    "metric_signal_coverage_up_0_65": 0.0022525506823903,
    "metric_signal_coverage_down_0_65": 0.0033788260235855,
    "metric_signal_coverage_up_0_7": 6.625149065853981e-05,
    "metric_signal_coverage_down_0_7": 0.0001987544719756,
    "metric_mean_forward_return_up_0_5": -6.428070513929818e-05,
    "metric_mean_forward_return_down_0_5": -8.386460959540833e-05,
    "metric_mean_forward_return_up_0_55": -0.0001069794964537,
    "metric_mean_forward_return_down_0_55": -0.0002180458587159,
    "metric_mean_forward_return_up_0_6": 5.988801095596237e-05,
    "metric_mean_forward_return_down_0_6": -0.0002345227739576,
    "metric_mean_forward_return_up_0_65": 0.0042038815534691,
    "metric_mean_forward_return_down_0_65": -0.0010681292107454,
    "metric_mean_forward_return_up_0_7": 0.0203265053082941,
    "metric_mean_forward_return_down_0_7": 0.0013475166862683
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 23

- asset: ETH/USDT
- timeframe: 1h
- horizon: 2h
- model: hist_gradient_boosting
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5456229159299268,
    "metric_pr_auc": 0.5405384575745381,
    "metric_balanced_accuracy": 0.5391738450144123,
    "metric_log_loss": 0.6910082298336983,
    "metric_brier_score": 0.2489175386709944,
    "metric_accuracy": 0.5394977168949772,
    "metric_mean_forward_return": -0.0001732699240616,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.5031963470319635,
    "metric_signal_coverage_up_0_5": 0.5509132420091324,
    "metric_signal_coverage_down_0_5": 0.4490867579908675,
    "metric_signal_coverage_up_0_55": 0.2482876712328767,
    "metric_signal_coverage_down_0_55": 0.1747716894977169,
    "metric_signal_coverage_up_0_6": 0.0541095890410958,
    "metric_signal_coverage_down_0_6": 0.0366438356164383,
    "metric_signal_coverage_up_0_65": 0.0063926940639269,
    "metric_signal_coverage_down_0_65": 0.003310502283105,
    "metric_signal_coverage_up_0_7": 0.0007990867579908,
    "metric_signal_coverage_down_0_7": 0.0005707762557077,
    "metric_mean_forward_return_up_0_5": -8.855451476123827e-05,
    "metric_mean_forward_return_down_0_5": 0.0002771938094922,
    "metric_mean_forward_return_up_0_55": -5.40201469034512e-05,
    "metric_mean_forward_return_down_0_55": 0.0003907816814055,
    "metric_mean_forward_return_up_0_6": 0.0012082603344923,
    "metric_mean_forward_return_down_0_6": 1.1224026177130958e-05,
    "metric_mean_forward_return_up_0_65": 0.0036744169925639,
    "metric_mean_forward_return_down_0_65": 0.0003239944320505,
    "metric_mean_forward_return_up_0_7": 0.0341316738766631,
    "metric_mean_forward_return_down_0_7": 0.0029327204674262
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5543077801341889,
    "metric_pr_auc": 0.5527306783166236,
    "metric_balanced_accuracy": 0.541482921188549,
    "metric_log_loss": 0.6892502895996724,
    "metric_brier_score": 0.2480473603957905,
    "metric_accuracy": 0.5412718346843247,
    "metric_mean_forward_return": 0.0001718847908739,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5087338737298779,
    "metric_signal_coverage_up_0_5": 0.4886402557369563,
    "metric_signal_coverage_down_0_5": 0.5113597442630438,
    "metric_signal_coverage_up_0_55": 0.1743349697454047,
    "metric_signal_coverage_down_0_55": 0.1891768466719945,
    "metric_signal_coverage_up_0_6": 0.0256878639114054,
    "metric_signal_coverage_down_0_6": 0.0434981162233131,
    "metric_signal_coverage_up_0_65": 0.001826692544811,
    "metric_signal_coverage_down_0_65": 0.0053659093503824,
    "metric_signal_coverage_up_0_7": 0.0002283365681013,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001429141337789,
    "metric_mean_forward_return_down_0_5": -0.0001995682944163,
    "metric_mean_forward_return_up_0_55": 0.0003922892722751,
    "metric_mean_forward_return_down_0_55": -0.0005131221033259,
    "metric_mean_forward_return_up_0_6": 0.0003933553151371,
    "metric_mean_forward_return_down_0_6": -0.0004834499061047,
    "metric_mean_forward_return_up_0_65": 0.0002579894927982,
    "metric_mean_forward_return_down_0_65": -0.0007142265088043,
    "metric_mean_forward_return_up_0_7": 0.0067801785017094,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5465781884934946,
    "metric_pr_auc": 0.5574580396003733,
    "metric_balanced_accuracy": 0.5312730096498757,
    "metric_log_loss": 0.6903343922836671,
    "metric_brier_score": 0.2485763966674938,
    "metric_accuracy": 0.5319899817850637,
    "metric_mean_forward_return": 0.0001328984479064,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5178734061930783,
    "metric_signal_coverage_up_0_5": 0.5211748633879781,
    "metric_signal_coverage_down_0_5": 0.4788251366120218,
    "metric_signal_coverage_up_0_55": 0.2122040072859745,
    "metric_signal_coverage_down_0_55": 0.1756602914389799,
    "metric_signal_coverage_up_0_6": 0.0264116575591985,
    "metric_signal_coverage_down_0_6": 0.0229963570127504,
    "metric_signal_coverage_up_0_65": 0.0028460837887067,
    "metric_signal_coverage_down_0_65": 0.0028460837887067,
    "metric_signal_coverage_up_0_7": 0.0003415300546448,
    "metric_signal_coverage_down_0_7": 0.0005692167577413,
    "metric_mean_forward_return_up_0_5": 5.285325749663057e-05,
    "metric_mean_forward_return_down_0_5": -0.0002200232414623,
    "metric_mean_forward_return_up_0_55": 0.0004555906709332,
    "metric_mean_forward_return_down_0_55": -3.280054659594324e-05,
    "metric_mean_forward_return_up_0_6": 0.0021995266948202,
    "metric_mean_forward_return_down_0_6": -0.0008080425759799,
    "metric_mean_forward_return_up_0_65": 0.0053554983416598,
    "metric_mean_forward_return_down_0_65": -0.0029138642953483,
    "metric_mean_forward_return_up_0_7": 0.0111942932080246,
    "metric_mean_forward_return_down_0_7": -0.0014984862584547
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5347440950102917,
    "metric_pr_auc": 0.5357069222120822,
    "metric_balanced_accuracy": 0.5234704649780565,
    "metric_log_loss": 0.6931245882661798,
    "metric_brier_score": 0.2499530541293148,
    "metric_accuracy": 0.5242827800967336,
    "metric_mean_forward_return": 2.6709005323328585e-05,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5087789041277413,
    "metric_signal_coverage_up_0_5": 0.5466772676075001,
    "metric_signal_coverage_down_0_5": 0.4533227323924998,
    "metric_signal_coverage_up_0_55": 0.2628370767905651,
    "metric_signal_coverage_down_0_55": 0.1769032001590141,
    "metric_signal_coverage_up_0_6": 0.0379646193599682,
    "metric_signal_coverage_down_0_6": 0.0314052872192407,
    "metric_signal_coverage_up_0_65": 0.003246538130259,
    "metric_signal_coverage_down_0_65": 0.0086795203074272,
    "metric_signal_coverage_up_0_7": 0.0004637911614655,
    "metric_signal_coverage_down_0_7": 0.0011263499635592,
    "metric_mean_forward_return_up_0_5": -9.30816342568372e-05,
    "metric_mean_forward_return_down_0_5": -0.0001711686029813,
    "metric_mean_forward_return_up_0_55": -8.954128595143808e-05,
    "metric_mean_forward_return_down_0_55": -0.0001953701539357,
    "metric_mean_forward_return_up_0_6": -5.183561752983682e-05,
    "metric_mean_forward_return_down_0_6": -0.0003058624725549,
    "metric_mean_forward_return_up_0_65": 0.0011591404240658,
    "metric_mean_forward_return_down_0_65": -0.0010101439015828,
    "metric_mean_forward_return_up_0_7": 0.0039433287118995,
    "metric_mean_forward_return_down_0_7": 0.0019699141025545
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 24

- asset: ETH/USDT
- timeframe: 1h
- horizon: 2h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.54007341473057,
    "metric_pr_auc": 0.5322192277688748,
    "metric_balanced_accuracy": 0.5329602093786698,
    "metric_log_loss": 0.6917890979581592,
    "metric_brier_score": 0.2493031602742874,
    "metric_accuracy": 0.5327625570776255,
    "metric_mean_forward_return": -0.0001732699240616,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.5031963470319635,
    "metric_signal_coverage_up_0_5": 0.4692922374429223,
    "metric_signal_coverage_down_0_5": 0.5307077625570776,
    "metric_signal_coverage_up_0_55": 0.1373287671232876,
    "metric_signal_coverage_down_0_55": 0.1470319634703196,
    "metric_signal_coverage_up_0_6": 0.0245433789954337,
    "metric_signal_coverage_down_0_6": 0.0159817351598173,
    "metric_signal_coverage_up_0_65": 0.0029680365296803,
    "metric_signal_coverage_down_0_65": 0.0013698630136986,
    "metric_signal_coverage_up_0_7": 0.0006849315068493,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0001224061310875,
    "metric_mean_forward_return_down_0_5": 0.0002182475650418,
    "metric_mean_forward_return_up_0_55": 0.0002028803305175,
    "metric_mean_forward_return_down_0_55": -0.0002012013036296,
    "metric_mean_forward_return_up_0_6": -0.0001343242302065,
    "metric_mean_forward_return_down_0_6": 0.001715773457904,
    "metric_mean_forward_return_up_0_65": 0.0033671856586311,
    "metric_mean_forward_return_down_0_65": 0.006321834066439,
    "metric_mean_forward_return_up_0_7": 0.0255170135116538,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5557144904540317,
    "metric_pr_auc": 0.5575181923271978,
    "metric_balanced_accuracy": 0.538615912826048,
    "metric_log_loss": 0.6890716117821588,
    "metric_brier_score": 0.2479711478497863,
    "metric_accuracy": 0.5370476081744492,
    "metric_mean_forward_return": 0.0001718847908739,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5087338737298779,
    "metric_signal_coverage_up_0_5": 0.4108916542984359,
    "metric_signal_coverage_down_0_5": 0.5891083457015641,
    "metric_signal_coverage_up_0_55": 0.0504623815504053,
    "metric_signal_coverage_down_0_55": 0.1234159150587966,
    "metric_signal_coverage_up_0_6": 0.0023975339650645,
    "metric_signal_coverage_down_0_6": 0.0023975339650645,
    "metric_signal_coverage_up_0_65": 0.0001141682840506,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 5.303646704183319e-05,
    "metric_mean_forward_return_down_0_5": -0.0002547791934847,
    "metric_mean_forward_return_up_0_55": 0.000146071744402,
    "metric_mean_forward_return_down_0_55": -0.0005897621775808,
    "metric_mean_forward_return_up_0_6": 0.0080410917108029,
    "metric_mean_forward_return_down_0_6": -0.0020440815446452,
    "metric_mean_forward_return_up_0_65": 0.0268873650710326,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5504839212427294,
    "metric_pr_auc": 0.5586591537860988,
    "metric_balanced_accuracy": 0.5385586774783202,
    "metric_log_loss": 0.6900584218006053,
    "metric_brier_score": 0.2484529877431005,
    "metric_accuracy": 0.5364298724954463,
    "metric_mean_forward_return": 0.0001328984479064,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5178734061930783,
    "metric_signal_coverage_up_0_5": 0.4418260473588342,
    "metric_signal_coverage_down_0_5": 0.5581739526411658,
    "metric_signal_coverage_up_0_55": 0.1040528233151183,
    "metric_signal_coverage_down_0_55": 0.1550546448087431,
    "metric_signal_coverage_up_0_6": 0.013091985428051,
    "metric_signal_coverage_down_0_6": 0.0071721311475409,
    "metric_signal_coverage_up_0_65": 0.0009107468123861,
    "metric_signal_coverage_down_0_65": 0.0003415300546448,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.000301565683889,
    "metric_mean_forward_return_down_0_5": 6.111468005873909e-07,
    "metric_mean_forward_return_up_0_55": 0.0005779013862848,
    "metric_mean_forward_return_down_0_55": 0.000175044787198,
    "metric_mean_forward_return_up_0_6": 0.0004015302979802,
    "metric_mean_forward_return_down_0_6": -0.0014967545521587,
    "metric_mean_forward_return_up_0_65": 0.0064246299292684,
    "metric_mean_forward_return_down_0_65": -0.0158775538429488,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5317210819497877,
    "metric_pr_auc": 0.5296988838483463,
    "metric_balanced_accuracy": 0.5215762174685756,
    "metric_log_loss": 0.6926483538247961,
    "metric_brier_score": 0.2497394705691211,
    "metric_accuracy": 0.5211024978466839,
    "metric_mean_forward_return": 2.6709005323328585e-05,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5087789041277413,
    "metric_signal_coverage_up_0_5": 0.4733982640959385,
    "metric_signal_coverage_down_0_5": 0.5266017359040615,
    "metric_signal_coverage_up_0_55": 0.1178692108924667,
    "metric_signal_coverage_down_0_55": 0.139137348439674,
    "metric_signal_coverage_up_0_6": 0.0126548731199894,
    "metric_signal_coverage_down_0_6": 0.0073544027032399,
    "metric_signal_coverage_up_0_65": 0.0009938382031405,
    "metric_signal_coverage_down_0_65": 0.0002650235208374,
    "metric_signal_coverage_up_0_7": 0.0001325117604187,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 1.8065679256942083e-06,
    "metric_mean_forward_return_down_0_5": -4.909550698489094e-05,
    "metric_mean_forward_return_up_0_55": -0.0003129647223471,
    "metric_mean_forward_return_down_0_55": -0.0001589683725639,
    "metric_mean_forward_return_up_0_6": -0.0009688214848991,
    "metric_mean_forward_return_down_0_6": 0.0002236135620008,
    "metric_mean_forward_return_up_0_65": 0.0089385119792848,
    "metric_mean_forward_return_down_0_65": 0.0028683809298521,
    "metric_mean_forward_return_up_0_7": 0.0377202994143127,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 25

- asset: ETH/USDT
- timeframe: 1h
- horizon: 2h
- model: random_forest
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.547719418487109,
    "metric_pr_auc": 0.5422960226939024,
    "metric_balanced_accuracy": 0.5376800703266787,
    "metric_log_loss": 0.6901911284319482,
    "metric_brier_score": 0.2485262849660571,
    "metric_accuracy": 0.5376712328767124,
    "metric_mean_forward_return": -0.0001732699240616,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.5031963470319635,
    "metric_signal_coverage_up_0_5": 0.4988584474885845,
    "metric_signal_coverage_down_0_5": 0.5011415525114156,
    "metric_signal_coverage_up_0_55": 0.2045662100456621,
    "metric_signal_coverage_down_0_55": 0.2214611872146118,
    "metric_signal_coverage_up_0_6": 0.0331050228310502,
    "metric_signal_coverage_down_0_6": 0.0275114155251141,
    "metric_signal_coverage_up_0_65": 0.0045662100456621,
    "metric_signal_coverage_down_0_65": 0.0031963470319634,
    "metric_signal_coverage_up_0_7": 0.0004566210045662,
    "metric_signal_coverage_down_0_7": 0.0005707762557077,
    "metric_mean_forward_return_up_0_5": -6.939486764075352e-05,
    "metric_mean_forward_return_down_0_5": 0.0002766717456013,
    "metric_mean_forward_return_up_0_55": 0.0002300273619985,
    "metric_mean_forward_return_down_0_55": 0.0004026386296161,
    "metric_mean_forward_return_up_0_6": -0.0004270464976137,
    "metric_mean_forward_return_down_0_6": 0.0011699005099901,
    "metric_mean_forward_return_up_0_65": 0.0060941883995489,
    "metric_mean_forward_return_down_0_65": 0.0054745470681975,
    "metric_mean_forward_return_up_0_7": 0.0250953206946586,
    "metric_mean_forward_return_down_0_7": -0.0030918848266159
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5650806334856355,
    "metric_pr_auc": 0.5624502457564362,
    "metric_balanced_accuracy": 0.55138470675755,
    "metric_log_loss": 0.6874725917485107,
    "metric_brier_score": 0.2471695161491483,
    "metric_accuracy": 0.5502911291243293,
    "metric_mean_forward_return": 0.0001718847908739,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5087338737298779,
    "metric_signal_coverage_up_0_5": 0.4382920424706016,
    "metric_signal_coverage_down_0_5": 0.5617079575293983,
    "metric_signal_coverage_up_0_55": 0.1294668341134832,
    "metric_signal_coverage_down_0_55": 0.2370133576892339,
    "metric_signal_coverage_up_0_6": 0.0073067701792442,
    "metric_signal_coverage_down_0_6": 0.0345929900673592,
    "metric_signal_coverage_up_0_65": 0.0010275145564562,
    "metric_signal_coverage_down_0_65": 0.0013700194086082,
    "metric_signal_coverage_up_0_7": 0.0001141682840506,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001898557790697,
    "metric_mean_forward_return_down_0_5": -0.0001578623063853,
    "metric_mean_forward_return_up_0_55": 0.0004086318917148,
    "metric_mean_forward_return_down_0_55": -0.0002447382084129,
    "metric_mean_forward_return_up_0_6": 0.0018762486891351,
    "metric_mean_forward_return_down_0_6": -0.0003122778176294,
    "metric_mean_forward_return_up_0_65": 0.0027616532428142,
    "metric_mean_forward_return_down_0_65": 0.0003024339498393,
    "metric_mean_forward_return_up_0_7": 0.0175530665387173,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5457881034611185,
    "metric_pr_auc": 0.5591574849792686,
    "metric_balanced_accuracy": 0.5354309352990382,
    "metric_log_loss": 0.6909434303959705,
    "metric_brier_score": 0.2488794377380873,
    "metric_accuracy": 0.5346083788706739,
    "metric_mean_forward_return": 0.0001328984479064,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5178734061930783,
    "metric_signal_coverage_up_0_5": 0.4782559198542805,
    "metric_signal_coverage_down_0_5": 0.5217440801457195,
    "metric_signal_coverage_up_0_55": 0.1937613843351548,
    "metric_signal_coverage_down_0_55": 0.2462431693989071,
    "metric_signal_coverage_up_0_6": 0.0149134790528233,
    "metric_signal_coverage_down_0_6": 0.0286885245901639,
    "metric_signal_coverage_up_0_65": 0.0015938069216757,
    "metric_signal_coverage_down_0_65": 0.0036429872495446,
    "metric_signal_coverage_up_0_7": 0.0003415300546448,
    "metric_signal_coverage_down_0_7": 0.0005692167577413,
    "metric_mean_forward_return_up_0_5": 0.0001733340435753,
    "metric_mean_forward_return_down_0_5": -9.5833220456055e-05,
    "metric_mean_forward_return_up_0_55": 0.0005545972703243,
    "metric_mean_forward_return_down_0_55": -0.0002946161579042,
    "metric_mean_forward_return_up_0_6": 0.0025939103354528,
    "metric_mean_forward_return_down_0_6": -0.0006959776697587,
    "metric_mean_forward_return_up_0_65": 0.0099840256836512,
    "metric_mean_forward_return_down_0_65": -0.0012413357012465,
    "metric_mean_forward_return_up_0_7": 0.0111942932080246,
    "metric_mean_forward_return_down_0_7": -0.0012045381408639
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5350327107168669,
    "metric_pr_auc": 0.5357966048343726,
    "metric_balanced_accuracy": 0.5283360850905463,
    "metric_log_loss": 0.6933697552912815,
    "metric_brier_score": 0.2500755069875742,
    "metric_accuracy": 0.5282581329092957,
    "metric_mean_forward_return": 2.6709005323328585e-05,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5087789041277413,
    "metric_signal_coverage_up_0_5": 0.4960577751275425,
    "metric_signal_coverage_down_0_5": 0.5039422248724574,
    "metric_signal_coverage_up_0_55": 0.2240773868680845,
    "metric_signal_coverage_down_0_55": 0.2490558537070165,
    "metric_signal_coverage_up_0_6": 0.0221957198701384,
    "metric_signal_coverage_down_0_6": 0.0349168488703372,
    "metric_signal_coverage_up_0_65": 0.0023189558073279,
    "metric_signal_coverage_down_0_65": 0.0079507056251242,
    "metric_signal_coverage_up_0_7": 0.0005300470416749,
    "metric_signal_coverage_down_0_7": 0.0004637911614655,
    "metric_mean_forward_return_up_0_5": -0.0001024695707509,
    "metric_mean_forward_return_down_0_5": -0.0001538665124319,
    "metric_mean_forward_return_up_0_55": -0.0001646737052833,
    "metric_mean_forward_return_down_0_55": -0.0002607210751703,
    "metric_mean_forward_return_up_0_6": -0.0001331127365727,
    "metric_mean_forward_return_down_0_6": -0.0004955965571227,
    "metric_mean_forward_return_up_0_65": 0.003222538108649,
    "metric_mean_forward_return_down_0_65": -0.0013597989104243,
    "metric_mean_forward_return_up_0_7": 0.0063135382312933,
    "metric_mean_forward_return_down_0_7": -0.0031196341884883
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 26

- asset: ETH/USDT
- timeframe: 1h
- horizon: 2h
- model: random_forest
- feature_set: crypto_core_v1+crypto_context_proxy_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5370075172480516,
    "metric_pr_auc": 0.5361666877414271,
    "metric_balanced_accuracy": 0.5223797223630832,
    "metric_log_loss": 0.6923967356756868,
    "metric_brier_score": 0.2496179712364264,
    "metric_accuracy": 0.5231735159817351,
    "metric_mean_forward_return": -0.0001732699240616,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.5031963470319635,
    "metric_signal_coverage_up_0_5": 0.6243150684931507,
    "metric_signal_coverage_down_0_5": 0.3756849315068493,
    "metric_signal_coverage_up_0_55": 0.3036529680365297,
    "metric_signal_coverage_down_0_55": 0.1350456621004566,
    "metric_signal_coverage_up_0_6": 0.0636986301369863,
    "metric_signal_coverage_down_0_6": 0.0121004566210045,
    "metric_signal_coverage_up_0_65": 0.0075342465753424,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0009132420091324,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.0002609809510952,
    "metric_mean_forward_return_down_0_5": 2.7511307578360213e-05,
    "metric_mean_forward_return_up_0_55": -0.0001282481090218,
    "metric_mean_forward_return_down_0_55": 4.809134557058088e-05,
    "metric_mean_forward_return_up_0_6": 0.0005531693212177,
    "metric_mean_forward_return_down_0_6": 0.000697397907037,
    "metric_mean_forward_return_up_0_65": 0.0077799195330875,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0198787372700672,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5600269591880076,
    "metric_pr_auc": 0.5533232017830619,
    "metric_balanced_accuracy": 0.5468425018493632,
    "metric_log_loss": 0.6880555205904968,
    "metric_brier_score": 0.2474618485686171,
    "metric_accuracy": 0.5461810708985044,
    "metric_mean_forward_return": 0.0001718847908739,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5087338737298779,
    "metric_signal_coverage_up_0_5": 0.4629523918255508,
    "metric_signal_coverage_down_0_5": 0.5370476081744492,
    "metric_signal_coverage_up_0_55": 0.1496746203904555,
    "metric_signal_coverage_down_0_55": 0.2012786847813677,
    "metric_signal_coverage_up_0_6": 0.0090192944400045,
    "metric_signal_coverage_down_0_6": 0.0245461810708985,
    "metric_signal_coverage_up_0_65": 0.0009133462724055,
    "metric_signal_coverage_down_0_65": 0.0005708414202534,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0001635470929731,
    "metric_mean_forward_return_down_0_5": -0.0001790721558798,
    "metric_mean_forward_return_up_0_55": 0.0005824526872703,
    "metric_mean_forward_return_down_0_55": -0.0002749948402229,
    "metric_mean_forward_return_up_0_6": 0.0001585645393674,
    "metric_mean_forward_return_down_0_6": -0.0001800962006374,
    "metric_mean_forward_return_up_0_65": 0.0008619176039487,
    "metric_mean_forward_return_down_0_65": 0.0002025332252859,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5425032370854629,
    "metric_pr_auc": 0.548805530329178,
    "metric_balanced_accuracy": 0.5312848964820427,
    "metric_log_loss": 0.6927436916363118,
    "metric_brier_score": 0.2497754728965694,
    "metric_accuracy": 0.5278916211293261,
    "metric_mean_forward_return": 0.0001328984479064,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5178734061930783,
    "metric_signal_coverage_up_0_5": 0.4061930783242258,
    "metric_signal_coverage_down_0_5": 0.5938069216757741,
    "metric_signal_coverage_up_0_55": 0.1035974499089253,
    "metric_signal_coverage_down_0_55": 0.2958788706739526,
    "metric_signal_coverage_up_0_6": 0.0064890710382513,
    "metric_signal_coverage_down_0_6": 0.0371129326047358,
    "metric_signal_coverage_up_0_65": 0.0006830601092896,
    "metric_signal_coverage_down_0_65": 0.0035291438979963,
    "metric_signal_coverage_up_0_7": 0.000455373406193,
    "metric_signal_coverage_down_0_7": 0.0006830601092896,
    "metric_mean_forward_return_up_0_5": 0.0001545457166001,
    "metric_mean_forward_return_down_0_5": -0.0001180906536773,
    "metric_mean_forward_return_up_0_55": 0.0005717629875386,
    "metric_mean_forward_return_down_0_55": -0.0002502792414898,
    "metric_mean_forward_return_up_0_6": 0.0033868015937094,
    "metric_mean_forward_return_down_0_6": -0.0004380005153846,
    "metric_mean_forward_return_up_0_65": 0.0142248756171528,
    "metric_mean_forward_return_down_0_65": 0.001096431593049,
    "metric_mean_forward_return_up_0_7": 0.0104521584529401,
    "metric_mean_forward_return_down_0_7": -0.0034953132064433
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5367270446661502,
    "metric_pr_auc": 0.5369137364964771,
    "metric_balanced_accuracy": 0.5287051299314309,
    "metric_log_loss": 0.6920616103353562,
    "metric_brier_score": 0.249437938362774,
    "metric_accuracy": 0.5295169946332737,
    "metric_mean_forward_return": 2.6709005323328585e-05,
    "metric_sample_count": 15093.0,
    "metric_positive_rate": 0.5087789041277413,
    "metric_signal_coverage_up_0_5": 0.5467435234877095,
    "metric_signal_coverage_down_0_5": 0.4532564765122904,
    "metric_signal_coverage_up_0_55": 0.2089710461803485,
    "metric_signal_coverage_down_0_55": 0.1447690982574703,
    "metric_signal_coverage_up_0_6": 0.0127211290001987,
    "metric_signal_coverage_down_0_6": 0.0172927847346452,
    "metric_signal_coverage_up_0_65": 0.0015901411250248,
    "metric_signal_coverage_down_0_65": 0.0035778175313059,
    "metric_signal_coverage_up_0_7": 0.0001987676406281,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -4.860497636683029e-05,
    "metric_mean_forward_return_down_0_5": -0.0001175569773898,
    "metric_mean_forward_return_up_0_55": 0.0002246873613415,
    "metric_mean_forward_return_down_0_55": -0.0002641919956249,
    "metric_mean_forward_return_up_0_6": 0.0008293219839665,
    "metric_mean_forward_return_down_0_6": -0.0005292159585621,
    "metric_mean_forward_return_up_0_65": 0.003955477142126,
    "metric_mean_forward_return_down_0_65": -0.0020446672543977,
    "metric_mean_forward_return_up_0_7": 0.0001278064020788,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 27

- asset: ETH/USDT
- timeframe: 1h
- horizon: 4h
- model: hist_gradient_boosting
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2024

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5406183273230644,
    "metric_pr_auc": 0.5359964012807384,
    "metric_balanced_accuracy": 0.5255928460797583,
    "metric_log_loss": 0.6923764306936517,
    "metric_brier_score": 0.2495791688947685,
    "metric_accuracy": 0.5259132420091325,
    "metric_mean_forward_return": -0.0003506457187375,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.5019406392694064,
    "metric_signal_coverage_up_0_5": 0.582648401826484,
    "metric_signal_coverage_down_0_5": 0.4173515981735159,
    "metric_signal_coverage_up_0_55": 0.2955479452054794,
    "metric_signal_coverage_down_0_55": 0.1662100456621004,
    "metric_signal_coverage_up_0_6": 0.0933789954337899,
    "metric_signal_coverage_down_0_6": 0.0421232876712328,
    "metric_signal_coverage_up_0_65": 0.0191780821917808,
    "metric_signal_coverage_down_0_65": 0.0133561643835616,
    "metric_signal_coverage_up_0_7": 0.0017123287671232,
    "metric_signal_coverage_down_0_7": 0.0023972602739726,
    "metric_mean_forward_return_up_0_5": -0.0001061144171975,
    "metric_mean_forward_return_down_0_5": 0.0006920263979115,
    "metric_mean_forward_return_up_0_55": 0.0003307787224853,
    "metric_mean_forward_return_down_0_55": 0.0009879045862713,
    "metric_mean_forward_return_up_0_6": 0.0018574108131312,
    "metric_mean_forward_return_down_0_6": 0.0037367210763171,
    "metric_mean_forward_return_up_0_65": 0.0036184614102408,
    "metric_mean_forward_return_down_0_65": 0.0028170563847343,
    "metric_mean_forward_return_up_0_7": 0.0110920860526211,
    "metric_mean_forward_return_down_0_7": -0.0018520243726896
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5563801930284762,
    "metric_pr_auc": 0.5515309820846195,
    "metric_balanced_accuracy": 0.5427379968896924,
    "metric_log_loss": 0.689486599407168,
    "metric_brier_score": 0.2481420254072781,
    "metric_accuracy": 0.5427560223769837,
    "metric_mean_forward_return": 0.0003449149984233,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5070213494691175,
    "metric_signal_coverage_up_0_5": 0.5018837766868364,
    "metric_signal_coverage_down_0_5": 0.4981162233131636,
    "metric_signal_coverage_up_0_55": 0.1728507820527457,
    "metric_signal_coverage_down_0_55": 0.2217148076264413,
    "metric_signal_coverage_up_0_6": 0.0277428930243178,
    "metric_signal_coverage_down_0_6": 0.0675876241580089,
    "metric_signal_coverage_up_0_65": 0.0060509190546866,
    "metric_signal_coverage_down_0_65": 0.0208927959812764,
    "metric_signal_coverage_up_0_7": 0.0009133462724055,
    "metric_signal_coverage_down_0_7": 0.0037675533736727,
    "metric_mean_forward_return_up_0_5": 0.0005108337106988,
    "metric_mean_forward_return_down_0_5": -0.0001777413428736,
    "metric_mean_forward_return_up_0_55": 0.0010756637986464,
    "metric_mean_forward_return_down_0_55": -0.0003435787226198,
    "metric_mean_forward_return_up_0_6": 0.0007563019328376,
    "metric_mean_forward_return_down_0_6": -0.0003241902646672,
    "metric_mean_forward_return_up_0_65": 0.0022636801193779,
    "metric_mean_forward_return_down_0_65": -0.0001014213214625,
    "metric_mean_forward_return_up_0_7": 0.0028896339302437,
    "metric_mean_forward_return_down_0_7": 0.0012990608666018
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5337626953917611,
    "metric_pr_auc": 0.5435612207555987,
    "metric_balanced_accuracy": 0.521831341459525,
    "metric_log_loss": 0.6924654354766818,
    "metric_brier_score": 0.2496520888558386,
    "metric_accuracy": 0.5224271402550091,
    "metric_mean_forward_return": 0.0002638566796267,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5099043715846995,
    "metric_signal_coverage_up_0_5": 0.5305100182149363,
    "metric_signal_coverage_down_0_5": 0.4694899817850637,
    "metric_signal_coverage_up_0_55": 0.2221083788706739,
    "metric_signal_coverage_down_0_55": 0.1838570127504553,
    "metric_signal_coverage_up_0_6": 0.0351775956284153,
    "metric_signal_coverage_down_0_6": 0.0255009107468123,
    "metric_signal_coverage_up_0_65": 0.003415300546448,
    "metric_signal_coverage_down_0_65": 0.0040983606557377,
    "metric_signal_coverage_up_0_7": 0.0006830601092896,
    "metric_signal_coverage_down_0_7": 0.0011384335154826,
    "metric_mean_forward_return_up_0_5": 0.0002587501409631,
    "metric_mean_forward_return_down_0_5": -0.0002696269197268,
    "metric_mean_forward_return_up_0_55": 0.0008351491767604,
    "metric_mean_forward_return_down_0_55": -0.0006533322731741,
    "metric_mean_forward_return_up_0_6": 0.0020757647868666,
    "metric_mean_forward_return_down_0_6": 0.0006966619634165,
    "metric_mean_forward_return_up_0_65": 0.0073304870941477,
    "metric_mean_forward_return_down_0_65": 0.0016497332968554,
    "metric_mean_forward_return_up_0_7": 0.0105003254240883,
    "metric_mean_forward_return_down_0_7": 0.0001728802859393
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5210541371744747,
    "metric_pr_auc": 0.5217203122283838,
    "metric_balanced_accuracy": 0.5169822303537264,
    "metric_log_loss": 0.697636732863432,
    "metric_brier_score": 0.2521029579564617,
    "metric_accuracy": 0.5177920614936055,
    "metric_mean_forward_return": 5.304751658855979e-05,
    "metric_sample_count": 15091.0,
    "metric_positive_rate": 0.5077198330130541,
    "metric_signal_coverage_up_0_5": 0.5527135378702538,
    "metric_signal_coverage_down_0_5": 0.4472864621297462,
    "metric_signal_coverage_up_0_55": 0.2524683586243456,
    "metric_signal_coverage_down_0_55": 0.158306275263402,
    "metric_signal_coverage_up_0_6": 0.0516864356238817,
    "metric_signal_coverage_down_0_6": 0.0420780597707242,
    "metric_signal_coverage_up_0_65": 0.0051023789013319,
    "metric_signal_coverage_down_0_65": 0.0173613411967397,
    "metric_signal_coverage_up_0_7": 0.0010602345769001,
    "metric_signal_coverage_down_0_7": 0.0071565833940759,
    "metric_mean_forward_return_up_0_5": -0.000169352563719,
    "metric_mean_forward_return_down_0_5": -0.0003278681195287,
    "metric_mean_forward_return_up_0_55": -0.0001429209258324,
    "metric_mean_forward_return_down_0_55": -8.463033646756362e-05,
    "metric_mean_forward_return_up_0_6": -0.0008628367527821,
    "metric_mean_forward_return_down_0_6": -0.0001921397981024,
    "metric_mean_forward_return_up_0_65": 0.0027247534680021,
    "metric_mean_forward_return_down_0_65": -0.000541246949042,
    "metric_mean_forward_return_up_0_7": 0.0030807969702661,
    "metric_mean_forward_return_down_0_7": -0.000506095773784
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 28

- asset: ETH/USDT
- timeframe: 1h
- horizon: 4h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5311413179375369,
    "metric_pr_auc": 0.5268708285533894,
    "metric_balanced_accuracy": 0.5246979388307335,
    "metric_log_loss": 0.6926040729091444,
    "metric_brier_score": 0.2497136018817214,
    "metric_accuracy": 0.5245433789954338,
    "metric_mean_forward_return": -0.0003506457187375,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.5019406392694064,
    "metric_signal_coverage_up_0_5": 0.4602739726027397,
    "metric_signal_coverage_down_0_5": 0.5397260273972603,
    "metric_signal_coverage_up_0_55": 0.1341324200913242,
    "metric_signal_coverage_down_0_55": 0.1437214611872146,
    "metric_signal_coverage_up_0_6": 0.0287671232876712,
    "metric_signal_coverage_down_0_6": 0.0149543378995433,
    "metric_signal_coverage_up_0_65": 0.0051369863013698,
    "metric_signal_coverage_down_0_65": 0.0009132420091324,
    "metric_signal_coverage_up_0_7": 0.0010273972602739,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -0.000228075855097,
    "metric_mean_forward_return_down_0_5": 0.0004551723029588,
    "metric_mean_forward_return_up_0_55": 0.0003282581833427,
    "metric_mean_forward_return_down_0_55": 6.289189385891197e-05,
    "metric_mean_forward_return_up_0_6": -0.0009062604628614,
    "metric_mean_forward_return_down_0_6": 0.0043924418880428,
    "metric_mean_forward_return_up_0_65": 0.0035034333268238,
    "metric_mean_forward_return_down_0_65": 0.0050834109401879,
    "metric_mean_forward_return_up_0_7": 0.0193674894167678,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5609529877549496,
    "metric_pr_auc": 0.5642596347082782,
    "metric_balanced_accuracy": 0.5461280778847237,
    "metric_log_loss": 0.6892334144265653,
    "metric_brier_score": 0.2480481630760169,
    "metric_accuracy": 0.5438977052174906,
    "metric_mean_forward_return": 0.0003449149984233,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5070213494691175,
    "metric_signal_coverage_up_0_5": 0.341819842447768,
    "metric_signal_coverage_down_0_5": 0.658180157552232,
    "metric_signal_coverage_up_0_55": 0.0267153784678616,
    "metric_signal_coverage_down_0_55": 0.0957871903185295,
    "metric_signal_coverage_up_0_6": 0.0014841876926589,
    "metric_signal_coverage_down_0_6": 0.001826692544811,
    "metric_signal_coverage_up_0_65": 0.0,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0006671293357271,
    "metric_mean_forward_return_down_0_5": -0.0001775759306197,
    "metric_mean_forward_return_up_0_55": 0.003026486550244,
    "metric_mean_forward_return_down_0_55": -0.0008073123989928,
    "metric_mean_forward_return_up_0_6": 0.0090202549209697,
    "metric_mean_forward_return_down_0_6": -0.0095255775347255,
    "metric_mean_forward_return_up_0_65": NaN,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5423854617457284,
    "metric_pr_auc": 0.5515989386753372,
    "metric_balanced_accuracy": 0.5329739325524534,
    "metric_log_loss": 0.6903814791685545,
    "metric_brier_score": 0.2486302489802561,
    "metric_accuracy": 0.5314207650273224,
    "metric_mean_forward_return": 0.0002638566796267,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5099043715846995,
    "metric_signal_coverage_up_0_5": 0.4222449908925318,
    "metric_signal_coverage_down_0_5": 0.5777550091074681,
    "metric_signal_coverage_up_0_55": 0.0776411657559198,
    "metric_signal_coverage_down_0_55": 0.1400273224043715,
    "metric_signal_coverage_up_0_6": 0.0087659380692167,
    "metric_signal_coverage_down_0_6": 0.0060336976320582,
    "metric_signal_coverage_up_0_65": 0.0018214936247723,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.000455373406193,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.0004813401896713,
    "metric_mean_forward_return_down_0_5": -0.0001049115882464,
    "metric_mean_forward_return_up_0_55": 0.0014374918247899,
    "metric_mean_forward_return_down_0_55": -0.0002281138418226,
    "metric_mean_forward_return_up_0_6": 0.0067097280822194,
    "metric_mean_forward_return_down_0_6": -0.0009195883329985,
    "metric_mean_forward_return_up_0_65": 0.021649506265259,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": 0.0265736354131267,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5267550825444066,
    "metric_pr_auc": 0.5237455701916678,
    "metric_balanced_accuracy": 0.5184645128674659,
    "metric_log_loss": 0.6930927131938942,
    "metric_brier_score": 0.2499683443195716,
    "metric_accuracy": 0.517659532171493,
    "metric_mean_forward_return": 5.304751658855979e-05,
    "metric_sample_count": 15091.0,
    "metric_positive_rate": 0.5077198330130541,
    "metric_signal_coverage_up_0_5": 0.4481479027234775,
    "metric_signal_coverage_down_0_5": 0.5518520972765224,
    "metric_signal_coverage_up_0_55": 0.095553641243125,
    "metric_signal_coverage_down_0_55": 0.1161619508316215,
    "metric_signal_coverage_up_0_6": 0.011530051023789,
    "metric_signal_coverage_down_0_6": 0.0059638194950632,
    "metric_signal_coverage_up_0_65": 0.0010602345769001,
    "metric_signal_coverage_down_0_65": 0.0003313233052812,
    "metric_signal_coverage_up_0_7": 0.0001325293221125,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": -5.3085474303352594e-06,
    "metric_mean_forward_return_down_0_5": -0.0001004372933608,
    "metric_mean_forward_return_up_0_55": -0.0010486172596347,
    "metric_mean_forward_return_down_0_55": 0.0003591792734952,
    "metric_mean_forward_return_up_0_6": 0.0002851300160635,
    "metric_mean_forward_return_down_0_6": 0.0040788286028926,
    "metric_mean_forward_return_up_0_65": 0.0160484330490479,
    "metric_mean_forward_return_down_0_65": 0.014455071537896,
    "metric_mean_forward_return_up_0_7": 0.0462229264507616,
    "metric_mean_forward_return_down_0_7": NaN
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 29

- asset: ETH/USDT
- timeframe: 1h
- horizon: 4h
- model: random_forest
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: B
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5364999712522515,
    "metric_pr_auc": 0.5383902328926481,
    "metric_balanced_accuracy": 0.5279206057554608,
    "metric_log_loss": 0.692528284554725,
    "metric_brier_score": 0.2496793755770209,
    "metric_accuracy": 0.5279680365296804,
    "metric_mean_forward_return": -0.0003506457187375,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.5019406392694064,
    "metric_signal_coverage_up_0_5": 0.5123287671232877,
    "metric_signal_coverage_down_0_5": 0.4876712328767123,
    "metric_signal_coverage_up_0_55": 0.223972602739726,
    "metric_signal_coverage_down_0_55": 0.2265981735159817,
    "metric_signal_coverage_up_0_6": 0.0431506849315068,
    "metric_signal_coverage_down_0_6": 0.0342465753424657,
    "metric_signal_coverage_up_0_65": 0.008904109589041,
    "metric_signal_coverage_down_0_65": 0.0050228310502283,
    "metric_signal_coverage_up_0_7": 0.0011415525114155,
    "metric_signal_coverage_down_0_7": 0.0003424657534246,
    "metric_mean_forward_return_up_0_5": 2.8981715505513144e-05,
    "metric_mean_forward_return_down_0_5": 0.0007494677985322,
    "metric_mean_forward_return_up_0_55": 0.0003701233625701,
    "metric_mean_forward_return_down_0_55": 3.488870604193992e-05,
    "metric_mean_forward_return_up_0_6": 0.0042635628830801,
    "metric_mean_forward_return_down_0_6": 0.0015088010364077,
    "metric_mean_forward_return_up_0_65": 0.0093788738398118,
    "metric_mean_forward_return_down_0_65": 0.0023087129651742,
    "metric_mean_forward_return_up_0_7": 0.0158076128039509,
    "metric_mean_forward_return_down_0_7": 0.0065230713984923
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5696577191000654,
    "metric_pr_auc": 0.5611938255666225,
    "metric_balanced_accuracy": 0.5507526032999799,
    "metric_log_loss": 0.6875821686586178,
    "metric_brier_score": 0.2472075903635481,
    "metric_accuracy": 0.5492636145678731,
    "metric_mean_forward_return": 0.0003449149984233,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.5070213494691175,
    "metric_signal_coverage_up_0_5": 0.3946797579632378,
    "metric_signal_coverage_down_0_5": 0.6053202420367622,
    "metric_signal_coverage_up_0_55": 0.1093732161205617,
    "metric_signal_coverage_down_0_55": 0.2762872474026715,
    "metric_signal_coverage_up_0_6": 0.0125585112455759,
    "metric_signal_coverage_down_0_6": 0.0804886402557369,
    "metric_signal_coverage_up_0_65": 0.0027400388172165,
    "metric_signal_coverage_down_0_65": 0.0252311907752026,
    "metric_signal_coverage_up_0_7": 0.0002283365681013,
    "metric_signal_coverage_down_0_7": 0.0037675533736727,
    "metric_mean_forward_return_up_0_5": 0.000824972966226,
    "metric_mean_forward_return_down_0_5": -3.1908511306394014e-05,
    "metric_mean_forward_return_up_0_55": 0.0015636867177569,
    "metric_mean_forward_return_down_0_55": -0.0002953408202804,
    "metric_mean_forward_return_up_0_6": 0.0028401108736318,
    "metric_mean_forward_return_down_0_6": -0.0002690199307266,
    "metric_mean_forward_return_up_0_65": 0.0063027737665718,
    "metric_mean_forward_return_down_0_65": -0.000833402687602,
    "metric_mean_forward_return_up_0_7": 0.0090199654007955,
    "metric_mean_forward_return_down_0_7": -0.0005172137359404
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5392383970725172,
    "metric_pr_auc": 0.5456978629518796,
    "metric_balanced_accuracy": 0.5289085807325398,
    "metric_log_loss": 0.6921505042523499,
    "metric_brier_score": 0.2494564860419001,
    "metric_accuracy": 0.5283469945355191,
    "metric_mean_forward_return": 0.0002638566796267,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5099043715846995,
    "metric_signal_coverage_up_0_5": 0.4722222222222222,
    "metric_signal_coverage_down_0_5": 0.5277777777777778,
    "metric_signal_coverage_up_0_55": 0.1727003642987249,
    "metric_signal_coverage_down_0_55": 0.2375910746812386,
    "metric_signal_coverage_up_0_6": 0.0179872495446265,
    "metric_signal_coverage_down_0_6": 0.0372267759562841,
    "metric_signal_coverage_up_0_65": 0.0026183970856102,
    "metric_signal_coverage_down_0_65": 0.0036429872495446,
    "metric_signal_coverage_up_0_7": 0.0009107468123861,
    "metric_signal_coverage_down_0_7": 0.0002276867030965,
    "metric_mean_forward_return_up_0_5": 0.0003580450462235,
    "metric_mean_forward_return_down_0_5": -0.0001795828779349,
    "metric_mean_forward_return_up_0_55": 0.0008882375485339,
    "metric_mean_forward_return_down_0_55": -0.0003337808692626,
    "metric_mean_forward_return_up_0_6": 0.0053242248390384,
    "metric_mean_forward_return_down_0_6": -0.0002849537401737,
    "metric_mean_forward_return_up_0_65": 0.0059694871490887,
    "metric_mean_forward_return_down_0_65": 0.002884148045619,
    "metric_mean_forward_return_up_0_7": 0.0026663506868541,
    "metric_mean_forward_return_down_0_7": -0.0024157251964154
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5266798906090859,
    "metric_pr_auc": 0.5272086355092491,
    "metric_balanced_accuracy": 0.5226349773417536,
    "metric_log_loss": 0.6960220289105891,
    "metric_brier_score": 0.2513406250785411,
    "metric_accuracy": 0.5224305877675436,
    "metric_mean_forward_return": 5.304751658855979e-05,
    "metric_sample_count": 15091.0,
    "metric_positive_rate": 0.5077198330130541,
    "metric_signal_coverage_up_0_5": 0.4871115234245576,
    "metric_signal_coverage_down_0_5": 0.5128884765754423,
    "metric_signal_coverage_up_0_55": 0.1956795440991319,
    "metric_signal_coverage_down_0_55": 0.2245709363196607,
    "metric_signal_coverage_up_0_6": 0.0300178914584851,
    "metric_signal_coverage_down_0_6": 0.0362467695977735,
    "metric_signal_coverage_up_0_65": 0.0049698495792194,
    "metric_signal_coverage_down_0_65": 0.0139155788218143,
    "metric_signal_coverage_up_0_7": 0.0011264992379563,
    "metric_signal_coverage_down_0_7": 0.0027168511033066,
    "metric_mean_forward_return_up_0_5": 3.0465227350353128e-06,
    "metric_mean_forward_return_down_0_5": -0.0001005355405959,
    "metric_mean_forward_return_up_0_55": -3.965406063499458e-05,
    "metric_mean_forward_return_down_0_55": -6.580638385548269e-05,
    "metric_mean_forward_return_up_0_6": 0.0005886632011131,
    "metric_mean_forward_return_down_0_6": -0.000442505185821,
    "metric_mean_forward_return_up_0_65": 0.0003361890190653,
    "metric_mean_forward_return_down_0_65": -0.001464198964835,
    "metric_mean_forward_return_up_0_7": 0.0074372902055884,
    "metric_mean_forward_return_down_0_7": -0.0035735662296132
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is B; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 30

- asset: ETH/USDT
- timeframe: 1h
- horizon: 8h
- model: logistic_regression
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5354038291048759,
    "metric_pr_auc": 0.5181241969355534,
    "metric_balanced_accuracy": 0.5234897271351411,
    "metric_log_loss": 0.692659943889371,
    "metric_brier_score": 0.2497094336838066,
    "metric_accuracy": 0.5246575342465754,
    "metric_mean_forward_return": -0.0007119132900584,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4924657534246575,
    "metric_signal_coverage_up_0_5": 0.4221461187214612,
    "metric_signal_coverage_down_0_5": 0.5778538812785388,
    "metric_signal_coverage_up_0_55": 0.123972602739726,
    "metric_signal_coverage_down_0_55": 0.130593607305936,
    "metric_signal_coverage_up_0_6": 0.0352739726027397,
    "metric_signal_coverage_down_0_6": 0.0151826484018264,
    "metric_signal_coverage_up_0_65": 0.0101598173515981,
    "metric_signal_coverage_down_0_65": 0.0027397260273972,
    "metric_signal_coverage_up_0_7": 0.0025114155251141,
    "metric_signal_coverage_down_0_7": 0.0001141552511415,
    "metric_mean_forward_return_up_0_5": 0.0003903216510713,
    "metric_mean_forward_return_down_0_5": 0.0015171414236614,
    "metric_mean_forward_return_up_0_55": 0.0023835159165239,
    "metric_mean_forward_return_down_0_55": 0.0029117969195372,
    "metric_mean_forward_return_up_0_6": -0.0014723190927034,
    "metric_mean_forward_return_down_0_6": 0.0048898538068654,
    "metric_mean_forward_return_up_0_65": -0.0036112334733265,
    "metric_mean_forward_return_down_0_65": 0.0128289504882819,
    "metric_mean_forward_return_up_0_7": -0.0105777127561813,
    "metric_mean_forward_return_down_0_7": 0.0534071943467917
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5646614444637144,
    "metric_pr_auc": 0.5526565978736586,
    "metric_balanced_accuracy": 0.5357621534167163,
    "metric_log_loss": 0.6886832418951416,
    "metric_brier_score": 0.2477770413867875,
    "metric_accuracy": 0.5387601324352095,
    "metric_mean_forward_return": 0.0006874426387322,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.4946911747916429,
    "metric_signal_coverage_up_0_5": 0.2172622445484644,
    "metric_signal_coverage_down_0_5": 0.7827377554515356,
    "metric_signal_coverage_up_0_55": 0.022148647105834,
    "metric_signal_coverage_down_0_55": 0.068272633862313,
    "metric_signal_coverage_up_0_6": 0.0015983559767096,
    "metric_signal_coverage_down_0_6": 0.0017125242607603,
    "metric_signal_coverage_up_0_65": 0.0,
    "metric_signal_coverage_down_0_65": 0.0,
    "metric_signal_coverage_up_0_7": 0.0,
    "metric_signal_coverage_down_0_7": 0.0,
    "metric_mean_forward_return_up_0_5": 0.002395226969728,
    "metric_mean_forward_return_down_0_5": -0.0002134179039182,
    "metric_mean_forward_return_up_0_55": 0.0054468636182801,
    "metric_mean_forward_return_down_0_55": 0.0001170417595582,
    "metric_mean_forward_return_up_0_6": 0.0087533970722617,
    "metric_mean_forward_return_down_0_6": 0.00948011384595,
    "metric_mean_forward_return_up_0_65": NaN,
    "metric_mean_forward_return_down_0_65": NaN,
    "metric_mean_forward_return_up_0_7": NaN,
    "metric_mean_forward_return_down_0_7": NaN
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5368491164567419,
    "metric_pr_auc": 0.5559406435479717,
    "metric_balanced_accuracy": 0.5233029822007583,
    "metric_log_loss": 0.6913587101560233,
    "metric_brier_score": 0.2491204217358656,
    "metric_accuracy": 0.519011839708561,
    "metric_mean_forward_return": 0.0005307885425045,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5153688524590164,
    "metric_signal_coverage_up_0_5": 0.3611111111111111,
    "metric_signal_coverage_down_0_5": 0.6388888888888888,
    "metric_signal_coverage_up_0_55": 0.0658014571948998,
    "metric_signal_coverage_down_0_55": 0.1229508196721311,
    "metric_signal_coverage_up_0_6": 0.0109289617486338,
    "metric_signal_coverage_down_0_6": 0.0068306010928961,
    "metric_signal_coverage_up_0_65": 0.0022768670309653,
    "metric_signal_coverage_down_0_65": 0.0003415300546448,
    "metric_signal_coverage_up_0_7": 0.0012522768670309,
    "metric_signal_coverage_down_0_7": 0.0001138433515482,
    "metric_mean_forward_return_up_0_5": 0.0006321830034031,
    "metric_mean_forward_return_down_0_5": -0.0004734786298227,
    "metric_mean_forward_return_up_0_55": 0.0035005375136789,
    "metric_mean_forward_return_down_0_55": 0.0001138534015714,
    "metric_mean_forward_return_up_0_6": 0.007966849758087,
    "metric_mean_forward_return_down_0_6": -0.0047760816534325,
    "metric_mean_forward_return_up_0_65": 0.00789592791634,
    "metric_mean_forward_return_down_0_65": -0.0050863944275956,
    "metric_mean_forward_return_up_0_7": 0.0269097355107532,
    "metric_mean_forward_return_down_0_7": -0.0100105932203389
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.509262917071488,
    "metric_pr_auc": 0.5191348468089378,
    "metric_balanced_accuracy": 0.504197207640213,
    "metric_log_loss": 0.6955069871643283,
    "metric_brier_score": 0.2511719902203344,
    "metric_accuracy": 0.5016901968582224,
    "metric_mean_forward_return": 0.0001108625183927,
    "metric_sample_count": 15087.0,
    "metric_positive_rate": 0.5132233048319745,
    "metric_signal_coverage_up_0_5": 0.4053158348246835,
    "metric_signal_coverage_down_0_5": 0.5946841651753165,
    "metric_signal_coverage_up_0_55": 0.0908729369655995,
    "metric_signal_coverage_down_0_55": 0.1112878637237356,
    "metric_signal_coverage_up_0_6": 0.0141181149333863,
    "metric_signal_coverage_down_0_6": 0.0064293762842182,
    "metric_signal_coverage_up_0_65": 0.0028501358785709,
    "metric_signal_coverage_down_0_65": 0.0007953867568104,
    "metric_signal_coverage_up_0_7": 0.000662822297342,
    "metric_signal_coverage_down_0_7": 6.628222973420826e-05,
    "metric_mean_forward_return_up_0_5": -8.578815555768885e-06,
    "metric_mean_forward_return_down_0_5": -0.0001922695354563,
    "metric_mean_forward_return_up_0_55": -0.0012420990964207,
    "metric_mean_forward_return_down_0_55": 0.0011399225346169,
    "metric_mean_forward_return_up_0_6": 0.0022455719876424,
    "metric_mean_forward_return_down_0_6": 0.0061492945973349,
    "metric_mean_forward_return_up_0_65": 0.0078184800906161,
    "metric_mean_forward_return_down_0_65": 0.0423034107969921,
    "metric_mean_forward_return_up_0_7": 0.0038449775283197,
    "metric_mean_forward_return_down_0_7": 0.0389315324701408
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.

## Candidate 31

- asset: ETH/USDT
- timeframe: 1h
- horizon: 8h
- model: random_forest
- feature_set: crypto_core_v1
- Phase 4.1 classification: ROBUST CANDIDATE
- Temporal diagnosis: A
- Positive AUC WFs: 3; negative AUC WFs: 0
- Best WF: wf_2023; worst WF: wf_2022

WF diagnostics:

```json
[
  {
    "split_id": "wf_2022",
    "metric_roc_auc": 0.5356974676387604,
    "metric_pr_auc": 0.5315600642471978,
    "metric_balanced_accuracy": 0.5252587533167286,
    "metric_log_loss": 0.6928217627901195,
    "metric_brier_score": 0.2498242622653552,
    "metric_accuracy": 0.5253424657534247,
    "metric_mean_forward_return": -0.0007119132900584,
    "metric_sample_count": 8760.0,
    "metric_positive_rate": 0.4924657534246575,
    "metric_signal_coverage_up_0_5": 0.4940639269406393,
    "metric_signal_coverage_down_0_5": 0.5059360730593607,
    "metric_signal_coverage_up_0_55": 0.2178082191780821,
    "metric_signal_coverage_down_0_55": 0.2284246575342465,
    "metric_signal_coverage_up_0_6": 0.0618721461187214,
    "metric_signal_coverage_down_0_6": 0.0559360730593607,
    "metric_signal_coverage_up_0_65": 0.0154109589041095,
    "metric_signal_coverage_down_0_65": 0.015296803652968,
    "metric_signal_coverage_up_0_7": 0.0038812785388127,
    "metric_signal_coverage_down_0_7": 0.0036529680365296,
    "metric_mean_forward_return_up_0_5": -6.55951218930853e-05,
    "metric_mean_forward_return_down_0_5": 0.0013430651474184,
    "metric_mean_forward_return_up_0_55": 0.0013683003170719,
    "metric_mean_forward_return_down_0_55": 0.0016424560881856,
    "metric_mean_forward_return_up_0_6": 0.0060983062799621,
    "metric_mean_forward_return_down_0_6": 0.0032860465524766,
    "metric_mean_forward_return_up_0_65": 0.0111244208042964,
    "metric_mean_forward_return_down_0_65": 0.0058845787721153,
    "metric_mean_forward_return_up_0_7": 0.0234730853039912,
    "metric_mean_forward_return_down_0_7": 0.004456109563528
  },
  {
    "split_id": "wf_2023",
    "metric_roc_auc": 0.5596264713191641,
    "metric_pr_auc": 0.5382831610696404,
    "metric_balanced_accuracy": 0.5426826603888714,
    "metric_log_loss": 0.6917183649146436,
    "metric_brier_score": 0.2491212986838792,
    "metric_accuracy": 0.5440118735015412,
    "metric_mean_forward_return": 0.0006874426387322,
    "metric_sample_count": 8759.0,
    "metric_positive_rate": 0.4946911747916429,
    "metric_signal_coverage_up_0_5": 0.3743578034022148,
    "metric_signal_coverage_down_0_5": 0.6256421965977851,
    "metric_signal_coverage_up_0_55": 0.131407694942345,
    "metric_signal_coverage_down_0_55": 0.3203562050462381,
    "metric_signal_coverage_up_0_6": 0.0508048864025573,
    "metric_signal_coverage_down_0_6": 0.1335768923393081,
    "metric_signal_coverage_up_0_65": 0.0192944400045667,
    "metric_signal_coverage_down_0_65": 0.049434866993949,
    "metric_signal_coverage_up_0_7": 0.0020550291129124,
    "metric_signal_coverage_down_0_7": 0.0173535791757049,
    "metric_mean_forward_return_up_0_5": 0.0014186322625924,
    "metric_mean_forward_return_down_0_5": -0.0002499297232875,
    "metric_mean_forward_return_up_0_55": 0.0015992848986933,
    "metric_mean_forward_return_down_0_55": -0.0001687799689363,
    "metric_mean_forward_return_up_0_6": 0.0015184422091627,
    "metric_mean_forward_return_down_0_6": -0.0003450867033086,
    "metric_mean_forward_return_up_0_65": 0.0011240083480625,
    "metric_mean_forward_return_down_0_65": -0.0005854870173339,
    "metric_mean_forward_return_up_0_7": 0.0035024178980301,
    "metric_mean_forward_return_down_0_7": -0.001289407893199
  },
  {
    "split_id": "wf_2024",
    "metric_roc_auc": 0.5371982860231662,
    "metric_pr_auc": 0.5560442241816226,
    "metric_balanced_accuracy": 0.5271854893658954,
    "metric_log_loss": 0.6929116816976085,
    "metric_brier_score": 0.2498529481396644,
    "metric_accuracy": 0.525728597449909,
    "metric_mean_forward_return": 0.0005307885425045,
    "metric_sample_count": 8784.0,
    "metric_positive_rate": 0.5153688524590164,
    "metric_signal_coverage_up_0_5": 0.4534380692167577,
    "metric_signal_coverage_down_0_5": 0.5465619307832422,
    "metric_signal_coverage_up_0_55": 0.1491347905282331,
    "metric_signal_coverage_down_0_55": 0.2509107468123862,
    "metric_signal_coverage_up_0_6": 0.0253870673952641,
    "metric_signal_coverage_down_0_6": 0.0482695810564663,
    "metric_signal_coverage_up_0_65": 0.0055783242258652,
    "metric_signal_coverage_down_0_65": 0.0100182149362477,
    "metric_signal_coverage_up_0_7": 0.0009107468123861,
    "metric_signal_coverage_down_0_7": 0.0006830601092896,
    "metric_mean_forward_return_up_0_5": 0.001025067304863,
    "metric_mean_forward_return_down_0_5": -0.0001207255742742,
    "metric_mean_forward_return_up_0_55": 0.0024386292633194,
    "metric_mean_forward_return_down_0_55": -0.0004122668006005,
    "metric_mean_forward_return_up_0_6": 0.0057092168284956,
    "metric_mean_forward_return_down_0_6": -0.0020179444915374,
    "metric_mean_forward_return_up_0_65": 0.0154175313529873,
    "metric_mean_forward_return_down_0_65": -0.0057809656294821,
    "metric_mean_forward_return_up_0_7": 0.0327532801156923,
    "metric_mean_forward_return_down_0_7": -0.0112108911970045
  }
]
```

Holdout diagnostics:

```json
[
  {
    "split_id": "final_holdout",
    "metric_roc_auc": 0.5141854882208599,
    "metric_pr_auc": 0.5270064406228832,
    "metric_balanced_accuracy": 0.5132409021768767,
    "metric_log_loss": 0.6999510760256534,
    "metric_brier_score": 0.2532313716784077,
    "metric_accuracy": 0.5124279180751641,
    "metric_mean_forward_return": 0.0001108625183927,
    "metric_sample_count": 15087.0,
    "metric_positive_rate": 0.5132233048319745,
    "metric_signal_coverage_up_0_5": 0.4696095976668655,
    "metric_signal_coverage_down_0_5": 0.5303904023331345,
    "metric_signal_coverage_up_0_55": 0.1786968913634254,
    "metric_signal_coverage_down_0_55": 0.2423941141379996,
    "metric_signal_coverage_up_0_6": 0.0418240869622854,
    "metric_signal_coverage_down_0_6": 0.0525618081792271,
    "metric_signal_coverage_up_0_65": 0.0108040034466759,
    "metric_signal_coverage_down_0_65": 0.0200835156094651,
    "metric_signal_coverage_up_0_7": 0.001922184662292,
    "metric_signal_coverage_down_0_7": 0.0057002717571419,
    "metric_mean_forward_return_up_0_5": 0.0001440527774159,
    "metric_mean_forward_return_down_0_5": -8.147574193940637e-05,
    "metric_mean_forward_return_up_0_55": -2.008467578687198e-05,
    "metric_mean_forward_return_down_0_55": 7.882582280988433e-05,
    "metric_mean_forward_return_up_0_6": 0.002898835675585,
    "metric_mean_forward_return_down_0_6": -0.0006861977830946,
    "metric_mean_forward_return_up_0_65": 0.0053722624162637,
    "metric_mean_forward_return_down_0_65": -0.0031219002661397,
    "metric_mean_forward_return_up_0_7": 0.0070687735838592,
    "metric_mean_forward_return_down_0_7": -0.0061477383622181
  }
]
```

Main observation: all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category is A; this is descriptive and not a promotion signal.
Potential follow-up: Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment.
