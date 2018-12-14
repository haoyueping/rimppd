"""
Query compiler convert a CQ into a label pattern or a list of label patterns.

Variables in CQ
    - head variables
    - session variables
    - other variables

Input
    a string of CQ

Output
    a label pattern / a list of label pattern
"""


class Compiler(object):
    pref_delimiter = ';'

    def __init__(self, query_str):
        self.query_str = query_str

    def compile(self, query_str):
        pass
