# -*- coding: utf-8 -*-
"""
Active learning: explore -> select -> (DFT in TFlow) -> import -> train, round after round.

    <campaign>/
      campaign.yaml           settings (edit freely between rounds)
      state.json              round, stage, current committee, dataset version
      ledger.jsonl            append-only: one record per step, never rewritten
      data/system.xyz         start structures for exploration
      data/initial.xyz        labelled data you started with (optional)
      data/test.xyz           fixed test set (optional, never changes during the campaign)
      data/labelled/round_NN.xyz   frames labelled by DFT in round NN
      data/train_vNN.xyz      rebuilt from initial + all labelled rounds (never appended to)
      round_NN/explore/ select/ export/ train/
"""
