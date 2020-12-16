import argparse
import dataclasses
import itertools
import re
from pathlib import Path
from typing import ClassVar, Iterable, Iterator, List, Optional

import pandas as pd
from google.cloud import bigquery
from google.cloud import bigquery_storage
from typing.io import TextIO


@dataclasses.dataclass
class TableInfo:
    _t: bigquery.TableReference
    _schema: Optional[List[bigquery.SchemaField]] = None
    client: ClassVar[Optional[bigquery.Client]] = None
    _shard_suffix: ClassVar[str] = r'_\d{8}\Z'

    @property
    def project_id(self) -> str:
        return self._t.project

    @property
    def dataset_id(self) -> str:
        return self._t.dataset_id

    @property
    def name(self) -> str:
        return self._t.table_id

    @property
    def clear_name(self) -> str:
        return re.sub(self._shard_suffix, '', self._t.table_id)

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        if self._schema is None:
            self._schema = self.client.get_table(self._t).schema
        return self._schema

    def is_sharding(self) -> bool:
        return bool(re.search(self._shard_suffix, self.name))

    @property
    def path(self) -> Path:
        return Path('.', 'views', self._t.dataset_id, f'{self.clear_name}.view.lkml')

    def create_dir(self):
        if not self.path.parents[0].is_dir():
            self.path.parents[0].mkdir(parents=True)


def get_dataset(project_id: str, dataset_id: str) -> bigquery.Dataset:
    return bigquery.Dataset(bigquery.DatasetReference(project_id, dataset_id))


def get_table_refs(
        bq_client: bigquery.Client,
        bq_storage_client: bigquery_storage.BigQueryReadClient,
        dataset: bigquery.Dataset
):
    query = f"select table_id from {dataset.project}.{dataset.dataset_id}.__TABLES__"
    df: pd.DataFrame = bq_client.query(query).to_dataframe(bqstorage_client=bq_storage_client)
    for table_id in df.table_id:
        if table_id is not None:
            yield bigquery.TableReference(dataset.reference, table_id)


def get_tables_info(tables: Iterator[bigquery.TableReference]) -> Iterator[TableInfo]:
    return sorted([TableInfo(t) for t in tables], key=lambda x: (x.clear_name, x.name))


def filter_latest_tables_info(table_info: Iterable[TableInfo]) -> Iterator[TableInfo]:
    for _, group in itertools.groupby(table_info, key=lambda x: x.clear_name):
        yield list(group)[-1]


def write_field(f: TextIO, field: bigquery.SchemaField):
    if field.field_type in ['TIME', 'TIMESTAMP', 'DATE', 'DATETIME']:
        f.write(f'  dimension_group: {field.name} {{\n')
    else:
        f.write(f'  dimension: {field.name} {{\n')
        if field.name == 'id':
            f.write('    primary_key: yes\n')

    if field.field_type in ['INTEGER', 'FLOAT', 'NUMERIC']:
        f.write('    type: number\n')
    elif field.field_type == 'BOOLEAN':
        f.write('    type: yesno\n')
    elif field.field_type in ['TIME', 'TIMESTAMP', 'DATE', 'DATETIME']:
        f.write('    type: time\n')
        f.write('    timeframes: [\n')
        f.write('      raw,\n')
        if field.field_type != 'DATE':
            f.write('      time,\n')
        f.write('      date,\n')
        f.write('      week,\n')
        f.write('      month,\n')
        f.write('      quarter,\n')
        f.write('      year\n')
        f.write('    ]\n')
        if field.field_type == 'DATE':
            f.write('    convert_tz: no\n')
            f.write('    datatype: date\n')
    elif field.field_type == 'RECORD':
        f.write('    hidden: yes\n')
    else:
        f.write('    type: string\n')

    f.write(f'    sql: ${{TABLE}}.{field.name} ;;\n')
    f.write('  }\n\n')


def write_record_child(f: TextIO, field: bigquery.SchemaField, prefix: str):
    f.write(f'view: {prefix}{field.name} {{\n')
    for fld in field.fields:
        write_field(f, fld)
    f.write('}\n\n')

    for fld in filter(lambda x: x.field_type == 'RECORD', field.fields):
        write_record_child(f, fld, f'{prefix}{field.name}__')


def write_look_ml(f: TextIO, info: TableInfo):
    # write view
    f.write(f'view: {info.dataset_id}__{info.clear_name} {{\n')
    f.write(f'  sql_table_name: `{info.project_id}.{info.dataset_id}.{info.clear_name}')
    if info.is_sharding():
        f.write('_*')
    f.write('`\n    ;;\n\n')

    for field in info.schema:
        write_field(f, field)

    # measure count
    f.write('  measure: count {\n')
    f.write('    type: count\n')

    drill_fields: List[str] = []
    if 'id' in [field.name for field in info.schema]:
        drill_fields.append('id')
    if 'name' in [field.name for field in info.schema]:
        drill_fields.append('name')

    f.write(f'    drill_fields: [{", ".join(drill_fields)}]\n')
    f.write('  }\n')
    f.write('}\n\n')

    for field in filter(lambda x: x.field_type == 'RECORD', info.schema):
        write_record_child(f, field, prefix=f'{info.dataset_id}__{info.clear_name}__')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_id')
    parser.add_argument('dataset_id')
    return parser.parse_args()


def main():
    args = parse_args()

    bq_client = bigquery.Client()
    bqs_client = bigquery_storage.BigQueryReadClient()

    TableInfo.client = bq_client

    dataset = get_dataset(args.project_id, args.dataset_id)

    table_refs = get_table_refs(bq_client, bqs_client, dataset)
    tables_info = get_tables_info(table_refs)
    tables_info = filter_latest_tables_info(tables_info)

    for info in tables_info:
        info.create_dir()
        with info.path.open('w') as f:
            print(f'write {info.clear_name}.view.lkml')
            write_look_ml(f, info)


if __name__ == '__main__':
    main()
