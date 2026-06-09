"""End-of-run federated learning + SIA metric summaries."""

import numpy as np


def print_federated_run_summary(
    trainacc,
    testacc,
    sia_accuracy_rounds,
    round_eod,
    round_loss_cov,
    round_loss_fi,
    round_sia_cov,
    round_sia_fi,
    round_average_loss_mad,
    round_sia_mad,
    round_sen_welfare,
    mia_attack_accuracy_rounds=None,
    mia_precision_rounds=None,
    mia_recall_rounds=None,
    mia_roc_auc_rounds=None,
    mia_client_mad_rounds=None,
    mia_reverse_sen_welfare_rounds=None,
    mia_cov_rounds=None,
    mia_fi_rounds=None,
    mia_eod_rounds=None,
    hessian_avg_time_rounds=None,
    nn_size_mib=None,
    *,
    last_k=3,
):
    """Per-round series plus aggregates (nan-safe)."""

    def _fmt_list(xs, nd=2):
        return '[' + ', '.join(
            f'{float(x):.{nd}f}' if np.isfinite(float(x)) else 'nan' for x in xs
        ) + ']'

    def _mean_last_k(xs, k):
        if not xs:
            return float('nan')
        sl = xs[-k:] if len(xs) >= k else xs
        return float(np.nanmean(np.asarray(sl, dtype=float)))

    print('\n' + '=' * 72)
    print('  RUN SUMMARY (per communication round, then aggregates)')
    print('=' * 72)
    print(f'  training accuracy (%):           {_fmt_list(trainacc, nd=2)}')
    print(f'  testing accuracy (%):            {_fmt_list(testacc, nd=2)}')
    print(
        f'  mean last {last_k} rounds — train %: {_mean_last_k(trainacc, last_k):.4f}  |  '
        f'test %: {_mean_last_k(testacc, last_k):.4f}'
    )

    print(f'  average SIA attack accuracy (%): {_fmt_list(sia_accuracy_rounds, nd=2)}')
    if sia_accuracy_rounds:
        arr = np.asarray(sia_accuracy_rounds, dtype=float)
        print(f'  mean of average SIA attack accuracy (%): {float(np.mean(arr)):.4f}')
        print(f'  max of average SIA attack accuracy (%):  {float(np.max(arr)):.4f}')
    else:
        print('  (no SIA accuracy rounds recorded)')

    if mia_attack_accuracy_rounds is not None:
        print(f'  MIA attack accuracy (per round):  {_fmt_list(mia_attack_accuracy_rounds, nd=4)}')
        if len(mia_attack_accuracy_rounds) > 0:
            mia_arr = np.asarray(mia_attack_accuracy_rounds, dtype=float)
            print(f'  mean MIA attack accuracy:        {float(np.nanmean(mia_arr)):.4f}')
            print(f'  max MIA attack accuracy:         {float(np.nanmax(mia_arr)):.4f}')
        else:
            print('  (no MIA accuracy rounds recorded)')
    if mia_precision_rounds is not None:
        print(f'  MIA precision (per round):        {_fmt_list(mia_precision_rounds, nd=4)}')
        if len(mia_precision_rounds) > 0:
            mia_prec_arr = np.asarray(mia_precision_rounds, dtype=float)
            print(f'  mean MIA precision:              {float(np.nanmean(mia_prec_arr)):.4f}')
        else:
            print('  (no MIA precision rounds recorded)')
    if mia_recall_rounds is not None:
        print(f'  MIA recall (per round):           {_fmt_list(mia_recall_rounds, nd=4)}')
        if len(mia_recall_rounds) > 0:
            mia_rec_arr = np.asarray(mia_recall_rounds, dtype=float)
            print(f'  mean MIA recall:                 {float(np.nanmean(mia_rec_arr)):.4f}')
        else:
            print('  (no MIA recall rounds recorded)')
    if mia_roc_auc_rounds is not None:
        print(f'  MIA ROC-AUC (per round):          {_fmt_list(mia_roc_auc_rounds, nd=4)}')
        if len(mia_roc_auc_rounds) > 0:
            mia_auc_arr = np.asarray(mia_roc_auc_rounds, dtype=float)
            print(f'  mean MIA ROC-AUC:                {float(np.nanmean(mia_auc_arr)):.4f}')
        else:
            print('  (no MIA ROC-AUC rounds recorded)')
    if mia_client_mad_rounds is not None:
        print(f'  MIA client MAD (per round):       {_fmt_list(mia_client_mad_rounds, nd=5)}')
        if len(mia_client_mad_rounds) > 0:
            mia_mad_arr = np.asarray(mia_client_mad_rounds, dtype=float)
            print(f'  mean MIA client MAD (across rounds): {float(np.nanmean(mia_mad_arr)):.5f}')
        else:
            print('  (no MIA MAD rounds recorded)')
    if mia_reverse_sen_welfare_rounds is not None:
        print(f'  reverse_mia sen_welfare (per round): {_fmt_list(mia_reverse_sen_welfare_rounds, nd=5)}')
        if len(mia_reverse_sen_welfare_rounds) > 0:
            mia_sw_arr = np.asarray(mia_reverse_sen_welfare_rounds, dtype=float)
            print(f'  mean reverse_mia sen_welfare (across rounds): {float(np.nanmean(mia_sw_arr)):.5f}')
        else:
            print('  (no reverse_mia sen_welfare rounds recorded)')
    if mia_cov_rounds is not None:
        print(f'  MIA CoV (per round):              {_fmt_list(mia_cov_rounds, nd=4)}')
        if len(mia_cov_rounds) > 0:
            print(f'  mean MIA CoV:                    {float(np.nanmean(np.asarray(mia_cov_rounds, dtype=float))):.4f}')
    if mia_fi_rounds is not None:
        print(f'  MIA FI (per round):               {_fmt_list(mia_fi_rounds, nd=4)}')
        if len(mia_fi_rounds) > 0:
            print(f'  mean MIA FI:                     {float(np.nanmean(np.asarray(mia_fi_rounds, dtype=float))):.4f}')
    if mia_eod_rounds is not None:
        print(f'  MIA EOD (per round):              {_fmt_list(mia_eod_rounds, nd=4)}')
        if len(mia_eod_rounds) > 0:
            print(f'  mean MIA EOD:                    {float(np.nanmean(np.asarray(mia_eod_rounds, dtype=float))):.4f}')
    if hessian_avg_time_rounds is not None:
        print(f'  Hessian avg time/client per round (s): {_fmt_list(hessian_avg_time_rounds, nd=4)}')
        if len(hessian_avg_time_rounds) > 0:
            hs_arr = np.asarray(hessian_avg_time_rounds, dtype=float)
            print(f'  mean Hessian avg time/client (s):      {float(np.nanmean(hs_arr)):.4f}')
        else:
            print('  (no Hessian timing rounds recorded)')
    if nn_size_mib is not None:
        print(f'  NN size (MiB):                    {float(nn_size_mib):.3f}')

    print(f'  EOD (max - min sia_per_client):  {_fmt_list(round_eod, nd=4)}')
    if round_eod:
        print(f'  mean EOD:                        {float(np.nanmean(np.asarray(round_eod, dtype=float))):.4f}')

    print(f'  Loss CoV:                        {_fmt_list(round_loss_cov, nd=4)}')
    print(f'  mean Loss CoV:                   {float(np.nanmean(np.asarray(round_loss_cov, dtype=float))):.4f}')
    print(f'  Loss FI:                         {_fmt_list(round_loss_fi, nd=4)}')
    print(f'  mean Loss FI:                    {float(np.nanmean(np.asarray(round_loss_fi, dtype=float))):.4f}')

    print(f'  Sia CoV:                         {_fmt_list(round_sia_cov, nd=4)}')
    print(f'  mean Sia CoV:                    {float(np.nanmean(np.asarray(round_sia_cov, dtype=float))):.4f}')
    print(f'  Sia FI:                          {_fmt_list(round_sia_fi, nd=4)}')
    print(f'  mean Sia FI:                     {float(np.nanmean(np.asarray(round_sia_fi, dtype=float))):.4f}')

    print(f'  average_loss_mad:                {_fmt_list(round_average_loss_mad, nd=5)}')
    print(
        f'  mean average_loss_mad:          '
        f'{float(np.nanmean(np.asarray(round_average_loss_mad, dtype=float))):.5f}'
    )

    print(f'  sia_mad:                         {_fmt_list(round_sia_mad, nd=5)}')
    print(
        f'  mean sia_mad:                   '
        f'{float(np.nanmean(np.asarray(round_sia_mad, dtype=float))):.5f}'
    )

    print(f'  sen_welfare:                     {_fmt_list(round_sen_welfare, nd=5)}')
    print(
        f'  mean sen_welfare:                '
        f'{float(np.nanmean(np.asarray(round_sen_welfare, dtype=float))):.5f}'
    )
    print('=' * 72 + '\n')
