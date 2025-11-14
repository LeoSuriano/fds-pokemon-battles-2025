from pathlib import Path
from collections import Counter
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore")

species = {
    'alakazam': {'hp': 55, 'atk': 50, 'def': 45, 'spa': 135, 'spd': 135, 'spe': 120},
    'articuno': {'hp': 90, 'atk': 85, 'def': 100, 'spa': 125, 'spd': 125, 'spe': 85},
    'chansey': {'hp': 250, 'atk': 5, 'def': 5, 'spa': 105, 'spd': 105, 'spe': 50},
    'charizard': {'hp': 78, 'atk': 84, 'def': 78, 'spa': 85, 'spd': 85, 'spe': 100},
    'cloyster': {'hp': 50, 'atk': 95, 'def': 180, 'spa': 85, 'spd': 85, 'spe': 70},
    'dragonite': {'hp': 91, 'atk': 134, 'def': 95, 'spa': 100, 'spd': 100, 'spe': 80},
    'exeggutor': {'hp': 95, 'atk': 95, 'def': 85, 'spa': 125, 'spd': 125, 'spe': 55},
    'gengar': {'hp': 60, 'atk': 65, 'def': 60, 'spa': 130, 'spd': 130, 'spe': 110},
    'golem': {'hp': 80, 'atk': 110, 'def': 130, 'spa': 55, 'spd': 55, 'spe': 45},
    'jolteon': {'hp': 65, 'atk': 65, 'def': 60, 'spa': 110, 'spd': 110, 'spe': 130},
    'jynx': {'hp': 65, 'atk': 50, 'def': 35, 'spa': 95, 'spd': 95, 'spe': 95},
    'lapras': {'hp': 130, 'atk': 85, 'def': 80, 'spa': 95, 'spd': 95, 'spe': 60},
    'persian': {'hp': 65, 'atk': 70, 'def': 60, 'spa': 65, 'spd': 65, 'spe': 115},
    'rhydon': {'hp': 105, 'atk': 130, 'def': 120, 'spa': 45, 'spd': 45, 'spe': 40},
    'slowbro': {'hp': 95, 'atk': 75, 'def': 110, 'spa': 80, 'spd': 80, 'spe': 30},
    'snorlax': {'hp': 160, 'atk': 110, 'def': 65, 'spa': 65, 'spd': 65, 'spe': 30},
    'starmie': {'hp': 60, 'atk': 75, 'def': 85, 'spa': 100, 'spd': 100, 'spe': 115},
    'tauros': {'hp': 75, 'atk': 100, 'def': 95, 'spa': 70, 'spd': 70, 'spe': 110},
    'victreebel': {'hp': 80, 'atk': 105, 'def': 65, 'spa': 100, 'spd': 100, 'spe': 70},
    'zapdos': {'hp': 90, 'atk': 90, 'def': 85, 'spa': 125, 'spd': 125, 'spe': 100},
}

types_map = {
    "alakazam": ["notype", "psychic"],
    "articuno": ["flying", "ice"],
    "chansey": ["normal", "notype"],
    "charizard": ["fire", "flying"],
    "cloyster": ["ice", "water"],
    "dragonite": ["dragon", "flying"],
    "exeggutor": ["grass", "psychic"],
    "gengar": ["ghost", "poison"],
    "golem": ["ground", "rock"],
    "jolteon": ["electric", "notype"],
    "jynx": ["ice", "psychic"],
    "lapras": ["ice", "water"],
    "persian": ["normal", "notype"],
    "rhydon": ["ground", "rock"],
    "slowbro": ["psychic", "water"],
    "snorlax": ["normal", "notype"],
    "starmie": ["psychic", "water"],
    "tauros": ["normal", "notype"],
    "victreebel": ["grass", "poison"],
    "zapdos": ["electric", "flying"]
}

effectiveness = {
    'normal':   {'rock': 0.5, 'ghost': 0, 'notype': 1},
    'fire':     {'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'fire': 0.5, 'water': 0.5, 'dragon': 0.5},
    'water':    {'fire': 2, 'rock': 2, 'ground': 2, 'water': 0.5, 'grass': 0.5, 'dragon': 0.5},
    'electric': {'water': 2, 'flying': 2, 'ground': 0, 'electric': 0.5, 'grass': 0.5, 'dragon': 0.5},
    'grass':    {'water': 2, 'rock': 2, 'ground': 2, 'fire': 0.5, 'grass': 0.5, 'poison': 0.5, 'flying': 0.5, 'dragon': 0.5},
    'ice':      {'grass': 2, 'ground': 2, 'flying': 2, 'dragon': 2, 'fire': 0.5, 'ice': 0.5, 'water': 0.5},
    'poison':   {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5},
    'ground':   {'fire': 2, 'electric': 2, 'poison': 2, 'rock': 2, 'grass': 0.5, 'flying': 0},
    'flying':   {'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'electric': 0.5},
    'psychic':  {'poison': 2, 'fighting': 2, 'psychic': 0.5},
    'bug':      {'grass': 2, 'psychic': 2, 'poison': 0.5, 'fire': 0.5, 'flying': 0.5},
    'rock':     {'fire': 2, 'ice': 2, 'flying': 2, 'bug': 2, 'ground': 0.5},
    'ghost':    {'ghost': 2, 'psychic': 0},
    'dragon':   {'dragon': 2},
    'notype':   {}
}

def type_match(attacking_type, defending_type):
    v = effectiveness.get(attacking_type, {}).get(defending_type, 1)
    if v == 0:
        return -2
    if v < 1:
        return -1
    if v > 1:
        return 1
    return 0

MAP_STATUS = {'nostatus': 0, 'par': 1, 'brn': 1, 'psn': 1, 'tox': 2, 'frz': 2, 'slp': 3, 'fnt': 0}

RECOVERY_MOVES = {'recover', 'softboiled', 'rest'}
HIGH_CRIT = {'slash', 'crabhammer', 'razorleaf', 'karatechop'}
TRAPS = {'wrap', 'clamp', 'firespin'}


def build_features(dataset):
    rows = []

    for battle in tqdm(dataset, desc="Building features"):
        feats = {}

        timeline = battle.get('battle_timeline', []) or []
        den = len(timeline)

        p1n = [str(t['p1_pokemon_state'].get('name', '')).lower() for t in timeline] if den else []
        p2n = [str(t['p2_pokemon_state'].get('name', '')).lower() for t in timeline] if den else []
        p1s = [str(t['p1_pokemon_state'].get('status', 'nostatus')).lower() for t in timeline] if den else []
        p2s = [str(t['p2_pokemon_state'].get('status', 'nostatus')).lower() for t in timeline] if den else []
        p1hp = [float(t['p1_pokemon_state'].get('hp_pct', 0.0)) for t in timeline] if den else []
        p2hp = [float(t['p2_pokemon_state'].get('hp_pct', 0.0)) for t in timeline] if den else []
        md1 = [(t.get('p1_move_details') or {}) for t in timeline] if den else []
        md2 = [(t.get('p2_move_details') or {}) for t in timeline] if den else []
        eff1 = [set(t['p1_pokemon_state'].get('effects') or []) for t in timeline] if den else []
        eff2 = [set(t['p2_pokemon_state'].get('effects') or []) for t in timeline] if den else []

        hp_last_p1, hp_last_p2 = {}, {}
        st_last_p1, st_last_p2 = {}, {}
        for t in timeline:
            n1 = str(t['p1_pokemon_state'].get('name', '')).lower()
            n2 = str(t['p2_pokemon_state'].get('name', '')).lower()
            hp_last_p1[n1] = float(t['p1_pokemon_state'].get('hp_pct', 0.0))
            hp_last_p2[n2] = float(t['p2_pokemon_state'].get('hp_pct', 0.0))
            st_last_p1[n1] = str(t['p1_pokemon_state'].get('status', 'nostatus')).lower()
            st_last_p2[n2] = str(t['p2_pokemon_state'].get('status', 'nostatus')).lower()

        mean_hp_p1 = float(np.mean(list(hp_last_p1.values()))) if hp_last_p1 else 0.0
        mean_hp_p2 = float(np.mean(list(hp_last_p2.values()))) if hp_last_p2 else 0.0
        feats['hp_edge_final'] = float(mean_hp_p2 - mean_hp_p1)

        feats['p1_alive_final'] = int(sum(hp > 0 for hp in hp_last_p1.values()))

        p1_status_mean_final = float(np.mean([MAP_STATUS.get(s, 0) for s in st_last_p1.values()])) if st_last_p1 else 0.0
        p2_status_mean_final = float(np.mean([MAP_STATUS.get(s, 0) for s in st_last_p2.values()])) if st_last_p2 else 0.0
        feats['p1_status_mean_final'] = p1_status_mean_final
        feats['status_severity_gap_final'] = float(p2_status_mean_final - p1_status_mean_final)

        revealed_p1 = set(p1n)
        revealed_p2 = set(p2n)
        feats['revealed_count_diff'] = int(len(revealed_p1) - len(revealed_p2))

        p1_status_series = [MAP_STATUS.get(s, 0) for s in p1s]
        p2_status_series = [MAP_STATUS.get(s, 0) for s in p2s]
        feats['status_turns_advantage'] = int(sum(p1_status_series) - sum(p2_status_series))

        p1_psn_cnt = sum(1 for s in p1s if s in {'psn', 'tox'})
        p2_psn_cnt = sum(1 for s in p2s if s in {'psn', 'tox'})
        p1_tox_cnt = sum(1 for s in p1s if s == 'tox')
        p2_tox_cnt = sum(1 for s in p2s if s == 'tox')
        p1_tox_ratio = (p1_tox_cnt / p1_psn_cnt) if p1_psn_cnt else 0.0
        p2_tox_ratio = (p2_tox_cnt / p2_psn_cnt) if p2_psn_cnt else 0.0
        feats['tox_ratio_diff'] = float(p1_tox_ratio - p2_tox_ratio)

        hp_gap = (np.array(p2hp) - np.array(p1hp)) if den else np.array([])

        E = max(1, den // 3) if den else 1
        mid_start, mid_end = E, max(E * 2, min(den, E * 2))

        def switches(seq):
            idx = []
            for i in range(1, len(seq)):
                if seq[i] != seq[i - 1]:
                    idx.append(i)
            return idx

        sw1 = switches(p1n) if den else []
        sw2 = switches(p2n) if den else []

        forced_p1 = 0
        forced_p2 = 0
        if den and len(hp_gap) > 1:
            for i in sw1:
                if i - 1 >= 0 and float(hp_gap[i] - hp_gap[i - 1]) > 0:
                    forced_p1 += 1
            for i in sw2:
                if i - 1 >= 0 and float(hp_gap[i] - hp_gap[i - 1]) < 0:
                    forced_p2 += 1
        p1_forced_share = (forced_p1 / len(sw1)) if sw1 else 0.0
        p2_forced_share = (forced_p2 / len(sw2)) if sw2 else 0.0
        feats['forced_switch_share_diff'] = float(p2_forced_share - p1_forced_share)

        def entropy(counter):
            tot = float(sum(counter.values()))
            if tot <= 0:
                return 0.0
            p = [c / tot for c in counter.values()]
            return float(-sum(pi * np.log(max(pi, 1e-12)) for pi in p))

        c1 = Counter(p1n)
        c2 = Counter(p2n)
        feats['active_entropy_diff'] = float(entropy(c1) - entropy(c2))

        p1_loss = [max(0.0, p1hp[i - 1] - p1hp[i]) for i in range(1, den)] if den > 1 else []
        p2_loss = [max(0.0, p2hp[i - 1] - p2hp[i]) for i in range(1, den)] if den > 1 else []
        p1_heal = [max(0.0, p1hp[i] - p1hp[i - 1]) for i in range(1, den)] if den > 1 else []
        p2_heal = [max(0.0, p2hp[i] - p2hp[i - 1]) for i in range(1, den)] if den > 1 else []

        feats['p2_late_damage'] = float(sum(p2_loss[-E:]) if p2_loss else 0.0)

        p1_attacks = sum(1 for m in md1 if m.get('name'))
        p2_attacks = sum(1 for m in md2 if m.get('name'))
        feats['attacks_rate_diff'] = float((p1_attacks / den if den else 0.0) - (p2_attacks / den if den else 0.0))

        feats['bp_mean_p2'] = float(
            np.mean([float(m.get('base_power', 0.0)) for m in md2
                     if m.get('name') and m.get('base_power') is not None]) if p2_attacks else 0.0
        )

        if den and len(hp_gap):
            feats['hp_gap_peak'] = float(np.max(hp_gap))
            feats['hp_gap_peak_turn_share'] = float((int(np.argmax(hp_gap)) + 1) / den)
            feats['hp_gap_var'] = float(np.var(hp_gap))

            if len(hp_gap) >= 2 and np.var(hp_gap) > 0 and np.var(hp_gap[:-1]) > 0 and np.var(hp_gap[1:]) > 0:
                feats['hp_gap_autocorr'] = float(np.corrcoef(hp_gap[:-1], hp_gap[1:])[0, 1])
            else:
                feats['hp_gap_autocorr'] = 0.0

            sgn = np.sign(hp_gap)
            feats['hp_gap_sign_flips'] = int(
                sum(1 for a, b in zip(sgn, sgn[1:])
                    if a != 0 and b != 0 and a != b)
            )
        else:
            feats['hp_gap_peak'] = 0.0
            feats['hp_gap_peak_turn_share'] = 0.0
            feats['hp_gap_var'] = 0.0
            feats['hp_gap_autocorr'] = 0.0
            feats['hp_gap_sign_flips'] = 0

        if den:
            lead1 = p1n[0]
            lead2 = p2n[0]
            t1 = [t for t in types_map.get(lead1, ["notype", "notype"]) if t != 'notype']
            t2 = [t for t in types_map.get(lead2, ["notype", "notype"]) if t != 'notype']
            lead_edge = 0
            for a in t2:
                for b in t1:
                    lead_edge += type_match(a, b)
                    lead_edge -= type_match(b, a)
            feats['lead_type_edge'] = int(lead_edge)

            d1 = species.get(lead1, {}).get('def', 0)
            d2 = species.get(lead2, {}).get('def', 0)
            feats['lead_def_edge'] = float(d2 - d1)
        else:
            feats['lead_type_edge'] = 0
            feats['lead_def_edge'] = 0.0

        if den:
            last1 = p1n[-1]
            last2 = p2n[-1]
            tp1 = types_map.get(last1, ["notype", "notype"])
            tp2 = types_map.get(last2, ["notype", "notype"])
            scf = 0
            for a in tp1:
                for b in tp2:
                    scf += type_match(a, b)
                    scf -= type_match(b, a)
            feats['types_last_round'] = int(scf)
        else:
            feats['types_last_round'] = 0

        se1 = rs1 = im1 = act1 = 0
        se2 = rs2 = im2 = act2 = 0
        for i in range(den):
            m1 = md1[i]
            m2 = md2[i]
            if m1.get('name'):
                mv_t = str(m1.get('type', '') or '').lower()
                if mv_t:
                    on = p2n[i] if i < len(p2n) else ''
                    for ot in [x for x in types_map.get(on, ["notype", "notype"]) if x != 'notype']:
                        v = type_match(mv_t, ot)
                        if v > 0:
                            se1 += 1
                        elif v < 0 and v != -2:
                            rs1 += 1
                        elif v == -2:
                            im1 += 1
                    act1 += 1
            if m2.get('name'):
                mv_t = str(m2.get('type', '') or '').lower()
                if mv_t:
                    on = p1n[i] if i < len(p1n) else ''
                    for ot in [x for x in types_map.get(on, ["notype", "notype"]) if x != 'notype']:
                        v = type_match(mv_t, ot)
                        if v > 0:
                            se2 += 1
                        elif v < 0 and v != -2:
                            rs2 += 1
                        elif v == -2:
                            im2 += 1
                    act2 += 1
        rs_share1 = (rs1 / act1) if act1 else 0.0
        rs_share2 = (rs2 / act2) if act2 else 0.0
        feats['rs_hit_share_diff'] = float(rs_share1 - rs_share2)
        feats['p1_immune_count'] = int(im1)
        feats['immune_count_diff'] = int(im1 - im2)

        feats['boom_count_diff'] = int(
            sum(1 for m in md1 if str(m.get('name', '')).lower() in {'explosion', 'selfdestruct'})
            - sum(1 for m in md2 if str(m.get('name', '')).lower() in {'explosion', 'selfdestruct'})
        )

        feats['counter_count_diff'] = int(
            sum(1 for m in md1 if str(m.get('name', '')).lower() == 'counter')
            - sum(1 for m in md2 if str(m.get('name', '')).lower() == 'counter')
        )

        feats['move_diversity_p1'] = int(len({str(m.get('name')) for m in md1 if m.get('name')}))

        c1 = Counter(p1n)
        c2 = Counter(p2n)

        def wmean(counter, key):
            S = 0.0
            W = 0.0
            for nm, w in counter.items():
                if nm in species:
                    S += float(species[nm].get(key, 0)) * w
                    W += w
            return (S / W) if W else 0.0

        p1_mean_spe_used = wmean(c1, 'spe')
        p2_mean_spe_used = wmean(c2, 'spe')
        feats['used_mean_spe_diff'] = float(p1_mean_spe_used - p2_mean_spe_used)
        feats['p2_used_count'] = int(len(c2))

        eff_adv = 0
        edge_sum = 0.0
        for i in range(den):
            s1 = timeline[i]['p1_pokemon_state']
            s2 = timeline[i]['p2_pokemon_state']
            n1 = str(s1.get('name', '')).lower()
            n2 = str(s2.get('name', '')).lower()
            base1 = species.get(n1, {})
            base2 = species.get(n2, {})
            b1 = (s1.get('boosts') or {})
            b2 = (s2.get('boosts') or {})

            try:
                sp1 = float(base1.get('spe', 0)) * (
                    (2.0 + int(b1.get('spe', 0))) / 2.0 if int(b1.get('spe', 0)) >= 0
                    else 2.0 / (2.0 - int(b1.get('spe', 0)))
                )
            except Exception:
                sp1 = float(base1.get('spe', 0))
            try:
                sp2 = float(base2.get('spe', 0)) * (
                    (2.0 + int(b2.get('spe', 0))) / 2.0 if int(b2.get('spe', 0)) >= 0
                    else 2.0 / (2.0 - int(b2.get('spe', 0)))
                )
            except Exception:
                sp2 = float(base2.get('spe', 0))

            if str(s1.get('status', '')).lower() == 'par':
                sp1 *= 0.25
            if str(s2.get('status', '')).lower() == 'par':
                sp2 *= 0.25

            if sp2 >= sp1:
                eff_adv += 1
            edge_sum += (sp2 - sp1)

        feats['eff_speed_adv_share_p2'] = float((eff_adv / den) if den else 0.0)
        feats['eff_speed_edge_avg'] = float((edge_sum / den) if den else 0.0)

        def init_share(seq_len, start, end):
            if end <= start or den == 0:
                return 0.0
            f = 0
            for i in range(start, end):
                if i >= den:
                    break
                n1 = p1n[i] if i < len(p1n) else ''
                n2 = p2n[i] if i < len(p2n) else ''
                s1 = species.get(n1, {}).get('spe', 0)
                s2 = species.get(n2, {}).get('spe', 0)
                if s1 >= s2:
                    f += 1
            L = max(1, end - start)
            return float(f / L) - float((L - f) / L)

        feats['initiative_early_diff'] = float(init_share(den, 0, min(E, den)))
        feats['initiative_late_diff'] = float(init_share(den, max(0, den - E), den))

        last_sw = 0
        for i in range(1, den):
            if p1n[i] != p1n[i - 1]:
                last_sw = i + 1
        feats['last_switch_turn_p1'] = int(last_sw if last_sw else den + 1)

        def pingpong(seq):
            c = 0
            for i in range(2, len(seq)):
                if seq[i] == seq[i - 2] and seq[i] != seq[i - 1]:
                    c += 1
            return c

        feats['p1_pingpong_switches'] = int(pingpong(p1n))
        feats['pingpong_switches_diff'] = int(pingpong(p1n) - pingpong(p2n))

        both_sw = 0
        for i in range(1, den):
            if p1n[i] != p1n[i - 1] and p2n[i] != p2n[i - 1]:
                both_sw += 1
        feats['both_switched_share'] = float(both_sw / max(1, (den - 1)))

        late_sw1 = 0
        for i in range(max(1, den - E), den):
            if i < len(p1n) and i - 1 >= 0 and p1n[i] != p1n[i - 1]:
                late_sw1 += 1
        feats['p1_switch_late_share'] = float(late_sw1 / max(1, min(E, den - 1)))

        severe = [(a >= 2 or b >= 2) for a, b in zip(p1_status_series, p2_status_series)]
        feats['severe_status_early_share'] = float(
            sum(1 for x in severe[:E] if x) / max(1, E)
        ) if den else 0.0

        sd1 = len({s for s in p1s if s != 'nostatus'})
        sd2 = len({s for s in p2s if s != 'nostatus'})
        feats['status_diversity_p1'] = int(sd1)
        feats['status_diversity_diff'] = int(sd1 - sd2)

        def share_status_late(md):
            late = md[max(0, den - E):den]
            tot = sum(1 for m in late if m.get('name'))
            sts = sum(1 for m in late if str(m.get('category', '')).upper() == 'STATUS' and m.get('name'))
            return (sts / tot) if tot else 0.0

        feats['status_late_share_diff'] = float(share_status_late(md2) - share_status_late(md1))

        def rec_share(md):
            tot = sum(1 for m in md if m.get('name'))
            rec = sum(1 for m in md if str(m.get('name', '')).lower() in RECOVERY_MOVES)
            return (rec / tot) if tot else 0.0

        feats['rec_share_diff'] = float(rec_share(md1) - rec_share(md2))

        def same_move_streak_max(md):
            best = cur = 0
            prev = None
            for m in md:
                nm = str(m.get('name', '')).lower() if m.get('name') else None
                if not nm:
                    cur = 0
                    prev = None
                else:
                    if nm == prev:
                        cur += 1
                    else:
                        cur = 1
                    best = max(best, cur)
                    prev = nm
            return best

        feats['same_move_streak_max_diff'] = int(same_move_streak_max(md1) - same_move_streak_max(md2))

        def eff_share(eff, tag, slc):
            seg = eff[slc]
            return float(sum(1 for s in seg if tag in s) / max(1, len(seg))) if den else 0.0

        early_slice = slice(0, min(E, den))
        late_slice = slice(max(0, den - E), den)
        feats['confusion_late_share_diff'] = float(
            eff_share(eff1, 'confusion', late_slice) - eff_share(eff2, 'confusion', late_slice)
        )
        feats['substitute_late_share_diff'] = float(
            eff_share(eff1, 'substitute', late_slice) - eff_share(eff2, 'substitute', late_slice)
        )
        feats['reflect_early_share_diff'] = float(
            eff_share(eff1, 'reflect', early_slice) - eff_share(eff2, 'reflect', early_slice)
        )

        feats['confusion_turns_diff'] = int(
            sum(1 for s in eff1 if 'confusion' in s) - sum(1 for s in eff2 if 'confusion' in s)
        )

        def max_streak(seq, tag):
            b = 0
            c = 0
            for s in seq:
                if s == tag:
                    c += 1
                    b = max(b, c)
                else:
                    c = 0
            return b

        feats['p1_sleep_streak_max'] = int(max_streak(p1s, 'slp'))
        feats['sleep_streak_max_diff'] = int(max_streak(p1s, 'slp') - max_streak(p2s, 'slp'))

        feats['p1_turns_par'] = int(sum(1 for s in p1s if s == 'par'))
        feats['p2_turns_brn'] = int(sum(1 for s in p2s if s == 'brn'))

        def seen_types(names):
            st = set()
            for nm in names:
                for tp in types_map.get(nm, ["notype", "notype"]):
                    if tp != 'notype':
                        st.add(tp)
            return st

        st1 = seen_types(revealed_p1)
        st2 = seen_types(revealed_p2)
        feats['type_seen_count_diff'] = int(len(st1) - len(st2))
        feats['p2_seen_type_count'] = int(len(st2))

        def types_of(name):
            return [x for x in types_map.get(name, ["notype", "notype"]) if x != 'notype']

        def type_multiplier(mv_type, opp_name):
            if not mv_type:
                return 1.0
            mult = 1.0
            for ot in types_of(opp_name):
                mult *= float(effectiveness.get(mv_type, {}).get(ot, 1.0))
            return float(mult)

        exp1 = []
        exp2 = []
        for i in range(den):
            m1 = md1[i]
            m2 = md2[i]
            if m1.get('name'):
                bp = float(m1.get('base_power', 0.0) or 0.0)
                acc = float(m1['accuracy']) if (m1.get('accuracy') is not None) else 1.0
                mv = str(m1.get('type', '') or '').lower()
                mon = p1n[i] if i < len(p1n) else ''
                stab = 1.5 if (mv and mv in types_of(mon)) else 1.0
                opp = p2n[i] if i < len(p2n) else ''
                tm = type_multiplier(mv, opp)
                exp1.append(bp * acc * stab * tm)
            else:
                exp1.append(0.0)
            if m2.get('name'):
                bp = float(m2.get('base_power', 0.0) or 0.0)
                acc = float(m2['accuracy']) if (m2.get('accuracy') is not None) else 1.0
                mv = str(m2.get('type', '') or '').lower()
                mon = p2n[i] if i < len(p2n) else ''
                stab = 1.5 if (mv and mv in types_of(mon)) else 1.0
                opp = p1n[i] if i < len(p1n) else ''
                tm = type_multiplier(mv, opp)
                exp2.append(bp * acc * stab * tm)
            else:
                exp2.append(0.0)

        exp1 = np.array(exp1) if den else np.array([])
        exp2 = np.array(exp2) if den else np.array([])
        feats['exp_dmg_stabtype_avg_diff'] = float(
            (np.mean(exp2) - np.mean(exp1)) if den else 0.0
        )

        def switch_delta(exp_vals, sw_idx):
            vals = []
            for i in sw_idx:
                prev = exp_vals[i - 1] if i - 1 >= 0 else 0.0
                vals.append(float(exp_vals[i] - prev))
            return float(np.mean(vals)) if vals else 0.0

        feats['switch_delta_exp_damage_diff'] = float(
            switch_delta(exp2, sw2) - switch_delta(exp1, sw1)
        )

        def ratio_when(eff, exp_vals, tag):
            with_tag = [exp_vals[i] for i, s in enumerate(eff) if tag in s]
            without = [exp_vals[i] for i, s in enumerate(eff) if tag not in s]
            mw = float(np.mean(with_tag)) if with_tag else 0.0
            mo = float(np.mean(without)) if without else 0.0
            return float(mw / (mo if mo != 0.0 else 1e-9))

        feats['confusion_exp_dmg_ratio_diff'] = float(
            ratio_when(eff2, exp2, 'confusion') - ratio_when(eff1, exp1, 'confusion')
        )

        def breaks(eff):
            return sum(1 for a, b in zip(eff, eff[1:]) if ('substitute' in a and 'substitute' not in b))

        p1_sub_t = sum(1 for s in eff1 if 'substitute' in s)
        p2_sub_t = sum(1 for s in eff2 if 'substitute' in s)
        p1_sub_b = breaks(eff1) if den else 0
        p2_sub_b = breaks(eff2) if den else 0
        r1 = (p1_sub_b / p1_sub_t) if p1_sub_t else 0.0
        r2 = (p2_sub_b / p2_sub_t) if p2_sub_t else 0.0
        feats['substitute_break_rate_diff'] = float(r1 - r2)

        p1_loss_sum = float(sum(p1_loss)) if p1_loss else 0.0
        p2_loss_sum = float(sum(p2_loss)) if p2_loss else 0.0
        p1_heal_sum = float(sum(p1_heal)) if p1_heal else 0.0
        p2_heal_sum = float(sum(p2_heal)) if p2_heal else 0.0
        p1_he_eff = float(p1_heal_sum / (p1_loss_sum if p1_loss_sum != 0.0 else 1e-9))
        p2_he_eff = float(p2_heal_sum / (p2_loss_sum if p2_loss_sum != 0.0 else 1e-9))
        feats['heal_efficiency_diff'] = float(p2_he_eff - p1_he_eff)
        feats['heal_mid_diff'] = float(
            (sum(p2_heal[E:mid_end]) - sum(p1_heal[E:mid_end])) if den > 1 else 0.0
        )
        feats['heal_late_diff'] = float(
            (sum(p2_heal[-E:]) - sum(p1_heal[-E:])) if den > 1 else 0.0
        )

        max_b2 = 0
        for t in timeline:
            b = (t['p2_pokemon_state'].get('boosts') or {})
            s = int(sum(int(v) for v in b.values())) if b else 0
            if s > max_b2:
                max_b2 = s
        feats['p2_max_boost_sum'] = int(max_b2)

        p1_team = battle.get('p1_team_details', []) or []
        if p1_team:
            p1_mean_atk_team = float(np.mean([int(p.get('base_atk', 0)) for p in p1_team]))
        else:
            p1_mean_atk_team = 0.0
        p2_mean_atk_used = wmean(c2, 'atk')
        feats['atk_edge_used'] = float(p2_mean_atk_used - p1_mean_atk_team)

        fb_p1 = None
        fb_p2 = None
        for i in range(1, len(p2s)):
            if p2s[i - 1] != 'fnt' and p2s[i] == 'fnt':
                fb_p1 = i + 1
                break
        for i in range(1, len(p1s)):
            if p1s[i - 1] != 'fnt' and p1s[i] == 'fnt':
                fb_p2 = i + 1
                break
        fb_turns = [x for x in (fb_p1, fb_p2) if x is not None]
        if den and len(hp_gap) >= 2 and fb_turns:
            fb = int(min(fb_turns))
            pre_s = max(0, fb - 1 - 5)
            pre_e = max(0, fb - 1)
            post_s = max(0, fb - 1)
            post_e = min(den, fb - 1 + 5)

            def slope(arr):
                if len(arr) < 2:
                    return 0.0
                x = np.arange(len(arr))
                try:
                    return float(np.polyfit(x, arr, 1)[0])
                except Exception:
                    return 0.0

            slope_pre = slope(hp_gap[pre_s:pre_e]) if pre_e - pre_s >= 2 else 0.0
            slope_post = slope(hp_gap[post_s:post_e]) if post_e - post_s >= 2 else 0.0
            feats['hp_gap_slope_jump'] = float(slope_post - slope_pre)
        else:
            feats['hp_gap_slope_jump'] = 0.0

        def sustained_share(arr, want_positive=True, min_len=3):
            if len(arr) == 0:
                return 0.0
            signs = np.sign(arr)
            target = 1 if want_positive else -1
            start = None
            for i in range(1, len(signs)):
                if (signs[i - 1] == -target) and (signs[i] == target):
                    start = i
                    break
            if start is None:
                return 0.0
            streak = 0
            for j in range(start, len(signs)):
                if signs[j] == target:
                    streak += 1
                    if streak >= min_len:
                        return float((len(signs) - j) / len(signs))
                else:
                    break
            return 0.0

        p1_cb = float(sustained_share(hp_gap, want_positive=False))
        p2_cb = float(sustained_share(-hp_gap, want_positive=False))
        feats['comeback_time_share_diff'] = float(p1_cb - p2_cb)

        feats['status_diversity_p1'] = int(sd1)
        feats['p2_seen_type_count'] = int(len(st2))
        feats['bp_mean_p2'] = feats['bp_mean_p2']

        feats['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            feats['player_won'] = int(battle['player_won'])

        rows.append(feats)

    df = pd.DataFrame(rows).fillna(0)
    return df

FEATURE_COLS_SUB2 = [
    "revealed_count_diff","hp_edge_final","status_severity_gap_final","p1_alive_final",
    "active_entropy_diff","status_turns_advantage","tox_ratio_diff","forced_switch_share_diff",
    "rs_hit_share_diff","boom_count_diff","counter_count_diff","move_diversity_p1",
    "used_mean_spe_diff","p2_late_damage","p1_status_mean_final","attacks_rate_diff",
    "bp_mean_p2","hp_gap_peak_turn_share","eff_speed_adv_share_p2","types_last_round",
    "atk_edge_used","last_switch_turn_p1","lead_type_edge","hp_gap_slope_jump",
    "initiative_late_diff","initiative_early_diff","comeback_time_share_diff",
    "severe_status_early_share","rec_share_diff","switch_delta_exp_damage_diff",
    "substitute_break_rate_diff","confusion_late_share_diff","status_diversity_p1",
    "hp_gap_autocorr","status_late_share_diff","pingpong_switches_diff",
    "same_move_streak_max_diff","substitute_late_share_diff","p1_sleep_streak_max",
    "immune_count_diff","heal_efficiency_diff","exp_dmg_stabtype_avg_diff","hp_gap_var",
    "both_switched_share","p1_switch_late_share","p2_max_boost_sum","status_diversity_diff",
    "hp_gap_sign_flips","reflect_early_share_diff","lead_def_edge","heal_mid_diff",
    "confusion_exp_dmg_ratio_diff","type_seen_count_diff","eff_speed_edge_avg",
    "heal_late_diff","sleep_streak_max_diff","p1_turns_par","p2_used_count",
    "hp_gap_peak","p2_turns_brn","p1_immune_count","p2_seen_type_count",
    "p1_pingpong_switches","confusion_turns_diff",
]

def create_features_sub2(train_data, test_data):
    train_df = build_features(train_data)
    test_df = build_features(test_data)

    X_train = train_df[FEATURE_COLS_SUB2].copy()
    y_train = train_df["player_won"].astype(int).copy()
    X_test = test_df[FEATURE_COLS_SUB2].copy()

    return X_train, y_train, X_test


