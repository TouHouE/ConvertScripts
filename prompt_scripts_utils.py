import os
import pandas as pd
import datetime as dt
import json
import numpy as np
import re
import traceback

EXPAND_SHORT_NAME = {
    'LM':"Left main coronary artery",
    "LAD": "Left anterior descending artery",
    "LCX": "Left circumflex artery",
    "RCA": "Right coronary artery"
}


def _extract_section(all_sections, section_name):
    for section in all_sections:
        if section_name in section:
            return section.split(section_name)[-1]
    return None


class CCTA:
    agatston: dict
    results: str
    coronary_analysis: dict
    noncoronary_analysis: list
    report_id: str
    check_date: dt.datetime | None
    dictionary: dict

    def __init__(self, df: pd.DataFrame, template_path: str):
        ccta_ctxt = df.get('CCTA 報告', df.get('Report'))
        # print(ccta_ctxt)
        pid = df.get('病歷號', df.get('病歷號碼', df.get('ReportID', None)))
        assert pid is not None
        # print(pid)
        check_date = df.get('檢查日期', None)
        self.template_path = template_path
        self.ccta_ctxt = ccta_ctxt
        self.report_id = str(pid).lower()
        self.check_date = check_date
        self.coronary_analysis: dict = dict()
        self.noncoronary_analysis: list = list()
        self.noncardiac_analysis: list = list()
        self.agatston: dict = dict()
        self.status = [False] * 4
        report_segment = re.split('\n[ ]{0,}\n', ccta_ctxt)

        # print(*report_segment, sep='|END_OF_SEG|\n')
        # self.results = report_segment[2].split('\n')[1:]
        if (section := _extract_section(report_segment, 'RESULTS:')) is not None:
            self.results = section.split("\n")[1:]
        else:
            self.results = []

        # Agatston Score.
        # If any issues arise in this section, I'll discard the current report.
        for line in _extract_section(report_segment, "Agatston score:").split('\n')[1:]:
            line = line.replace("\n", "")
            # print(line)
            key, value = re.split(':[ ]{0,}', line)
            key = re.split('[ ]{0,}\*[ ]{0,}', key)[-1]  # Remove list symbol, ex: "* "
            pure_digit = value.replace(' ', '')

            # Using regex to detect is a numerical string or not
            if re.fullmatch('[+-]{0,1}[0-9]{1,}\.{0,1}[0-9]{0,}', pure_digit) is not None: # Only has Agatston Score
                value = float(pure_digit)
                ps = ''
            else: # [Score] ([describe])
                # print(f"WTF, |{value}|, line:|{line}")
                value, ps = value.split('(')
                value = value.replace(' ', '')
                ps = ps.split(')')[0]

            self.agatston[key] = {'value': value, 'ps': ps}
            self.status[0] = True
        if 'Total' in self.agatston:
            del self.agatston['Total']

        # Process Coronary Analysis
        if (section := _extract_section(report_segment, "Coronary artery analysis:")) is not None:
            for line in section.split('\n')[1:]:
                line = line[2:]
                if ':' not in line:
                    self.coronary_analysis['desc'] = line
                    continue
                tag, desc = line.split(':')
                if desc.endswith('. '):
                    desc = desc[:-2]  # Remove ". "
                if 'Segment involvement score' in tag:
                    if 'None' not in desc:
                        score, desc = desc.split("(")
                        score = score.replace(" ", "")
                        desc, category = desc.split(")")[0].split(', ')
                        scale = desc.split(' ')[0]
                        self.coronary_analysis[tag] = {
                            "score": float(score),
                            "scale": scale,
                            'desc': desc,
                            "category": category  # like "P1", "P2" ...
                        }
                    continue
                    # If The "Segment involvement score" present "None. ", We just ignore this feature.

                self.coronary_analysis[tag] = desc[1:]
            self.status[1] = True
        # Process Non-Coronary cardiac Analysis
        if (section := _extract_section(report_segment, 'Noncoronary cardiac findings:')) is not None:
            for line in section.split('\n')[1:]:
                self.noncoronary_analysis.append(line[2:].replace('. ', ''))
            self.status[2] = True
        # Process Non-Cardiac findings
        if (section := _extract_section(report_segment, 'Noncardiac findings:')) is not None:
            for line in section.split('\n')[1:]:
                self.noncoronary_analysis.append(line[2:].replace('. ', ''))
            self.status[3] = True
        with open(template_path, 'r') as jin:
            self.dictionary = json.load(jin)

    def reload_template(self, new_template=None):
        if new_template is not None:
            if os.path.isfile(new_template):
                self.template_path = new_template
        with open(self.template_path, 'r', encoding='utf-8') as jin:
            self.dictionary = json.load(jin)

    def agatston_prompt(self, num_p=5):
        mapper = self.dictionary['agatston']
        if len(mapper) == 0:
            return [].copy()
        ask_l = np.random.choice(mapper['ask'], size=num_p, replace=True)
        ans_l = np.random.choice(mapper['ans'], size=num_p, replace=True)
        artery_keys = np.random.choice(list(self.agatston.keys()), size=num_p, replace=True)
        img_loc = np.random.randint(0, 2, num_p)
        prompt_list = []

        for ask, ans, artery, img_loc_ in zip(ask_l, ans_l, artery_keys, img_loc):
            score = self.agatston[artery]['value']
            full_name = EXPAND_SHORT_NAME[artery]
            ask = ask.replace('[Cardiac Name]', full_name.lower())
            ans = ans.replace('[Cardiac Name]', full_name.lower()).replace('[score]', str(score))
            ask = f'{ask}\n<image>' if img_loc_ == 1 else f'<image>\n{ask}'

            prompt_list.append({
                "pid": self.report_id,
                "date": self.check_date,
                "conversations": [
                    {
                        "from": "human",
                        "value": ask
                    },
                    {
                        "from": "gpt",
                        "value": ans
                    }
                ]
            })


        return prompt_list

    def coronary_prompt(self, n_p=5):
        def __filted_topic(_topic, _exist_topic):
            if isinstance(_topic, list):
                return True
            return _topic in _exist_topic
        prompt_list = []
        exist_topic = list(self.coronary_analysis.keys())
        if len(exist_topic) == 0:
            return prompt_list

        prompt_book = self.dictionary['coronary_analysis']
        all_pair = list(filter(lambda x: __filted_topic(x['topic'], exist_topic), prompt_book))
        # print(all_pair)

        for pair, img_loc in zip(np.random.choice(all_pair, n_p), np.random.randint(0, 2, n_p)):
            ask = pair['ask']
            ans = pair['ans']

            if isinstance(pair['topic'], list): # For Artery Prompt.
                artery = np.random.choice(pair['topic'], 1).tolist()[0]
                artery_lower = artery.lower()
                value = self.coronary_analysis.get(artery, 'Patent')
                ask = ask.replace('[artery]', artery_lower)

                if 'Patent' in value:
                    ans = ans['patent'].replace('[artery]', artery_lower)
                else:
                    ans = ans['ow'].replace('[artery]', artery_lower).replace('[describe]', value)
            elif pair['topic'] in 'Segment involvement score':
                info = self.coronary_analysis['Segment involvement score']
                ans = ans.replace('[score]', str(info.get('score', '')))
                ans = ans.replace('[category]', info.get('category', ''))
                ans = ans.replace('[scale]', info.get('scale', '').lower())
                ans = ans.replace('[describe]', info.get('describe', ''))
            elif pair['topic'] == 'Uninterpretable segments':
                value = self.coronary_analysis.get('Uninterpretable segments', 'none')
                if 'none' in value.lower():
                    ans = ans['none']
                else:
                    ans = ans['ow'].replace('[describe]', value)
            elif pair['topic'] == "Dominance":
                value = self.coronary_analysis['Dominance']
                ans = ans.replace('[describe]', value.lower())

            if img_loc == 0:
                ask = f'<image>\n{ask}'
            else:
                ask = f'{ask}\n<image>'

            prompt_list.append({
                "pid": self.report_id.lower(),
                "date": self.check_date,
                'conversations': [
                    {
                        'from': 'human',
                        'value': ask
                    },
                    {
                        'from': 'gpt',
                        'value': ans
                    }
                ]
            })

        return prompt_list

    def non_coronary_prompt(self, n_p):
        prompt_list = []
        if len(self.noncoronary_analysis) == 0:
            return prompt_list
        prompt_book = self.dictionary['noncoronary_analysis']
        ask_list = np.random.choice(prompt_book['ask'], n_p)
        ans_list = np.random.choice(prompt_book['ans'], n_p)
        loc_list = np.random.randint(0, 2, n_p)
        size_of_finding = len(self.noncoronary_analysis)
        num_finding = np.random.randint(1, size_of_finding + 1, n_p)
        selected_findings_list = [np.random.choice(self.noncoronary_analysis, num) for num in num_finding]

        for ask, ans, img_loc, selected_finding in zip(ask_list, ans_list, loc_list, selected_findings_list):
            if img_loc == 0:
                ask = f'<image>\n{ask}'
            else:
                ask = f'{ask}\n<image>'
            if len(selected_finding) > 2:
                finding_str = ','.join(selected_finding[:-1])
                finding_str = f'{finding_str} and {selected_finding[-1]}'
            elif len == 2:
                finding_str = f'{selected_finding[0]} and {selected_finding[1]}'
            else:
                finding_str = selected_finding[0]

            ans = ans.replace('[finding list]', finding_str)
            prompt_list.append({
                'pid': self.report_id,
                'date': self.check_date,
                "conversations": [
                    {
                        'from': 'human',
                        'value': ask
                    },
                    {
                        'from': 'gpt',
                        'value': ans
                    }
                ]
            })

        return prompt_list

    def non_cardiac_prompt(self, n_p):
        prompt_list = []
        if len(self.noncardiac_analysis) == 0:
            return prompt_list
        prompt_book = self.dictionary['noncardiac_analysis']
        pair_list = np.random.choice(prompt_book, n_p)
        loc_list = np.random.randint(0, 2, n_p)
        size_of_finding = len(self.noncardiac_analysis)
        num_finding = np.random.randint(1, size_of_finding + 1, n_p)
        selected_findings_list = [np.random.choice(self.noncardiac_analysis, num) for num in num_finding]

        for pair, img_loc, selected_finding in zip(pair_list, loc_list, selected_findings_list):
            ask = pair['ask']
            ans = pair['ans']
            if img_loc == 0:
                ask = f'<image>\n{ask}'
            else:
                ask = f'{ask}\n<image>'
            if len(selected_finding) > 2:
                finding_str = ','.join(selected_finding[:-1])
                finding_str = f'{finding_str} and {selected_finding[-1]}'
            elif len == 2:
                finding_str = f'{selected_finding[0]} and {selected_finding[1]}'
            else:
                finding_str = selected_finding[0]

            ans = ans.replace('[finding list]', finding_str)
            prompt_list.append({
                'pid': self.report_id,
                'date': self.check_date,
                "conversations": [
                    {
                        'from': 'human',
                        'value': ask
                    },
                    {
                        'from': 'gpt',
                        'value': ans
                    }
                ]
            })

        return prompt_list

    def get_prompt(self, n_prompt) -> list:
        size = 4
        target = n_prompt * size
        prompt_list = []

        for is_alive in enumerate(self.status):
            size -= int(not is_alive)
            n_prompt = target // size

        p0 = self.agatston_prompt(n_prompt)
        p1 = self.coronary_prompt(n_prompt * 2)
        p2 = self.non_coronary_prompt(n_prompt)
        p3 = self.non_cardiac_prompt(n_prompt)
        prompt_list.extend(p0)
        prompt_list.extend(p1)
        prompt_list.extend(p2)
        prompt_list.extend(p3)
        return prompt_list

    def __repr__(self):
        return self.ccta_ctxt


class NiiCTContainer:
    path: str
    date: list[dt.datetime]
    pid: str
    legal_path: dict[dt.datetime, list[str]]

    def __init__(self, pid: str, info_txt: pd.DataFrame):
        name_list: list[str] = info_txt[0].to_list()
        # The datetime store at idx -2
        date_list_cand: list[str] = [_name.split('_')[-2] for _name in name_list]
        # uni_date: set[dt.datetime] = set()
        date_mapper: dict[dt.datetime, list[str]] = dict()

        for date_cand, path in zip(date_list_cand, name_list):
            date = None
            try:
                date = dt.datetime.strptime(date_cand, '%Y%m%d%H%M%S')
            except ValueError as ve:
                continue

            if date_mapper.get(date) is None:
                date_mapper[date] = list()
            date_mapper[date].append(path)
        self.legal_path = date_mapper
        self.date = list(set(date_mapper.keys()))
        self.pid = pid

    def len(self, key: dt.datetime | str) -> int:
        if key is None:
            _tmp = list()
            for _comp in self.legal_path.values():
                _tmp.extend(_comp)
            return len(_tmp)

        if isinstance(key, str):
            key = dt.datetime.strptime(key, '%Y%m%d%H%M%S')
        return len(self.legal_path.get(key))

    def __getitem__(self, item):
        if item is None:
            _all_comp = list()
            for comp in list(self.legal_path.values()):
                _all_comp.extend(comp)
            return _all_comp
        return self.legal_path.get(item)

    def __eq__(self, other) -> bool:
        if isinstance(other, CCTA):
            same_patient = other.report_id.lower() == self.pid.lower()

            # If anyone didn't have member 'date', we ignore compare this feature
            if self.date is None or other.check_date is None:
                return same_patient

            return same_patient and other.check_date in self.date
        elif isinstance(other, NiiCTContainer):
            same_patient = other.pid.lower() == self.pid.lower()

            # If anyone didn't have member 'date', we ignore compare this feature
            if self.date is None or other.date is None:
                return same_patient
            return same_patient and self.date == other.date
        return False


def info_txt2CT(pid: str, info_txt: pd.DataFrame):
    return NiiCTContainer(pid, info_txt)


def build_ccta(df_entity: pd.DataFrame, template_path: str) -> CCTA:
    try:
        ccta = CCTA(df_entity, template_path=template_path)
    except Exception as e:
        print(traceback.format_exc())
        # print(df_entity)
        ccta = None

    return ccta
