# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： number_input_field.py
    @date：2023/10/31 17:58
    @desc:
"""
from typing import List

from common.froms.base_field import BaseField, TriggerType


class NumberInput(BaseField):
    """
    文本输入框
    """

    def __init__(self, label: str,
                 required: bool = False,
                 default_value=None,
                 relation_show_field_list: List[str] = None,
                 relation_show_value_list: List[str] = None,
                 attrs=None, props_info=None):
        super().__init__('NumberInput', label, required, default_value, relation_show_field_list,
                         relation_show_value_list, [], [],
                         TriggerType.OPTION_LIST, attrs, props_info)
