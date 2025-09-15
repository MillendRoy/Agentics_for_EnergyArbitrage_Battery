import re
import pandas as pd
import aiosqlite, asyncio, sqlite3
from typing import Set

def fix_double_quoted_literals(sql: str) -> str:
    """
    Convert double-quoted *literals* to single-quoted strings.
    Keep double-quoted *identifiers* like "MyTable" as-is.

    Heuristic: if the content is a simple identifier ([A-Za-z_][A-Za-z0-9_]*),
    we keep the double quotes; otherwise we treat it as a literal and convert.
    """
    ident_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

    def repl(m):
        body = m.group(1).replace('""', '"')  # unescape doubled quotes inside ""
        if ident_re.fullmatch(body):
            # looks like an identifier → leave as "Identifier"
            return f'"{m.group(1)}"'
        # looks like a literal → convert to '...'
        return "'" + body.replace("'", "''") + "'"

    # Match " ... " allowing doubled "" inside
    return re.sub(r'"((?:[^"]|"")*)"', repl, sql)



def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema_json={}
    for table in tables:

        cursor.execute(f"PRAGMA table_info({table[0]});")
        schema = cursor.fetchall()
        
        schema_json[table[0]] = { col[1] : {"type" : col[2], "notnull" : col[3], "dflt_value": col[4], } for col in schema} 
    return schema_json


def remove_duplicate_col_df(df):
    return df.loc[:, ~df.columns.duplicated()]


def convert_df_to_set(df, row_invariant=True) -> Set:
    # remove duplicate columns
    df = remove_duplicate_col_df(df)

    if row_invariant:
        return set(
            [
                tuple(sorted(df[c].to_list(), key=lambda x: (x is None, str(x))))
                for c in df.columns.values
            ]
        )
    else:
        return set([tuple(df[c].to_list()) for c in df.columns.values])



def compare_df(gt, predicted, row_invariant=False) -> bool:
    # 1: gt_df is subset of predicted_df
    # 2: df1 == df2
    # 0: otherwise
    if predicted.startswith("Error:") or gt.startswith("Error:"): return 0
    gt_df = pd.read_json(gt)
    predicted_df = pd.read_json(predicted)
    gt_df = gt_df.map(lambda x: float(f"{x:.5f}") if isinstance(x, float) else x)
    predicted_df = predicted_df.map(
        lambda x: float(f"{x:.5f}") if isinstance(x, float) else x
    )

    gt_set = convert_df_to_set(gt_df, row_invariant=row_invariant)
    predicted_set = convert_df_to_set(predicted_df, row_invariant=row_invariant)

    intersec = gt_set & predicted_set
    return (
        1
        if (intersec == gt_set)
        else 1
        if (predicted_set == gt_set)
        else 1
        if (intersec == predicted_set)
        else 0
    )

async def async_execute_sql(sql_query: str, db_path:str) -> str:
        try:
            async with aiosqlite.connect(db_path) as db:
                async with db.execute(sql_query.replace("\"","'")) as cursor:
                    columns = [description[0] for description in cursor.description]
                    rows = await asyncio.wait_for(cursor.fetchall(), timeout=10)
                    df = pd.DataFrame(rows, columns=columns)
                    return df.to_json()
        except Exception as e:
            return f"Error: {str(e)}"
        

def evaluate_execution_accuracy(test):
        total = 0
        for question in test:
            total+= compare_df(question.system_output_df, question.gt_output_df)
            print(question.gt_output_df, 
                  question.system_output_df, 
                  question.generated_query,
                  compare_df(question.system_output_df, question.gt_output_df) )
        execution_accuracy = total/len(test.states)
        print(f"Test size: {len(test.states)}\nExecution Accuracy: {execution_accuracy}")
        return execution_accuracy