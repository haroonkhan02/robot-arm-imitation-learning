#!/bin/sh
# Controls that separate memory, multi-step prediction and architecture, plus repeat seeds.
# Resumable: a finished model or evaluation is skipped. Rollouts run on the CPU, which is
# faster than MPS at batch size 1. Usage: scripts/run_controls.sh mlp|act|seeds|chunk5|last|decay|eval
set -e
EP=300

train() {  # name, ra-train arguments...
  name=$1; shift
  [ -f models/$name.json ] || ra-train --out models/$name.pt "$@" > outputs/train_$name.log 2>&1
}

score() {  # name, output tag, extra ra-policy-eval arguments...
  name=$1; tag=$2; shift 2
  [ -f reports/eval_$tag.json ] || ra-policy-eval --ckpt models/$name.pt --episodes $EP \
    --device cpu --out reports/eval_$tag.json "$@"
}

both() {  # score with temporal ensembling and with the newest chunk only
  score $1 $1
  score $1 $1_newest --ensemble -1000
}

sweep_epochs() { spe=$(($1 * 122 / 256)); [ $spe -lt 1 ] && spe=1; echo $((6000 / spe)); }

case "$1" in
mlp)
  for h in 2 4 8 12 20; do train bc_h$h --policy bc --epochs 100 --history $h; score bc_h$h bc_h$h; done
  train bc_big --policy bc --epochs 100 --hidden 1024 --layers 4; score bc_big bc_big
  for k in 5 20; do train bc_k$k --policy bc --epochs 100 --chunk $k; both bc_k$k; done
  train bc_h12_k20 --policy bc --epochs 100 --history 12 --chunk 20; both bc_h12_k20
  ;;
act)
  for k in 1 5 10; do train act_k$k --policy act --epochs 50 --chunk $k; done
  score act_k1 act_k1
  both act_k5; both act_k10
  ;;
seeds)
  for s in 1 2; do
    train bc_clean_s$s --policy bc --epochs 100 --seed $s; score bc_clean_s$s bc_clean_s$s
    train bc_k20_s$s --policy bc --epochs 100 --chunk 20 --seed $s; both bc_k20_s$s
    train act_clean_s$s --policy act --epochs 50 --seed $s; both act_clean_s$s
    for n in 10 20 50; do
      train act_n${n}_s$s --policy act --n-train-episodes $n --epochs $(sweep_epochs $n) --seed $s
      score act_n${n}_s$s act_n${n}_s$s
    done
  done
  ;;
chunk5)  # repeat seeds for the chunk size where MLP and ACT disagree most
  for s in 1 2; do
    train bc_k5_s$s --policy bc --epochs 100 --chunk 5 --seed $s; both bc_k5_s$s
    train act_k5_s$s --policy act --epochs 50 --chunk 5 --seed $s; both act_k5_s$s
  done
  ;;
last)  # ACT from the final epoch, against the best-validation checkpoints it usually keeps
  train act_last --policy act --epochs 50 --keep last; both act_last
  for s in 1 2; do train act_last_s$s --policy act --epochs 50 --keep last --seed $s; both act_last_s$s; done
  ;;
decay)  # how much weight older predictions need: larger m trusts the newest chunk more
  for m in 0.1 0.3 1; do
    for name in bc_k20 act_clean; do score $name ${name}_decay$m --ensemble $m; done
  done
  ;;
eval)  # earlier checkpoints, re-scored so that every run has stage data and 300 episodes
  for m in act_clean act_noisy bc_clean bc_noisy act_n10 act_n20 act_n50 act_n120 bc_n10 bc_n20 bc_n50 bc_n120; do
    rm -f reports/eval_$m.json; score $m $m
  done
  rm -f reports/eval_act_clean_newest.json
  score act_clean act_clean_newest --ensemble -1000
  for noise in 0.01 0.02 0.03 0.05; do
    score act_clean noise_act_clean_$noise --exec-noise $noise
    score act_noisy noise_act_noisy_$noise --exec-noise $noise
    ra-expert-eval --episodes $EP --exec-noise $noise --out reports/noise_expert_$noise.json
  done
  ;;
*) echo "usage: $0 mlp|act|seeds|chunk5|last|decay|eval" >&2; exit 2 ;;
esac
