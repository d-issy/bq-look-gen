import argparse
import dataclasses
import itertools
import re
from pathlib import Path
from typing import ClassVar, Iterable, Iterator, List
from typing.io import TextIO

from google.cloud import bigquery


@dataclasses.dataclass(frozen=True)
class TableInfo:
    _t: bigquery.Table
    _shard_suffix: ClassVar[str] = r'_\d{8}$'

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
        return self._t.schema

    def is_sharding(self) -> bool:
        return bool(re.match(self._shard_suffix, self.name))

    @property
    def path(self) -> Path:
        return Path('.', 'views', self._t.dataset_id, f'{self.clear_name}.view')

    def create_dir(self):
        if not self.path.parents[0].is_dir():
            self.path.parents[0].mkdir(parents=True)

    def cat(self):
        with self.path.open('r') as f:
            print(f.read())


def get_client() -> bigquery.Client:
    return bigquery.Client()


def get_dataset(project_id: str, dataset_id: str) -> bigquery.Dataset:
    return bigquery.Dataset(bigquery.DatasetReference(project_id, dataset_id))


def get_tables(client: bigquery.Client, dataset: bigquery.Dataset) -> Iterator[bigquery.Table]:
    tables: Iterator[bigquery.table.TableListItem] = client.list_tables(dataset)
    for t in tables:
        yield client.get_table(t.reference)


def get_tables_info(tables: Iterator[bigquery.Table]) -> Iterator[TableInfo]:
    return sorted([TableInfo(t) for t in tables], key=lambda x: (x.clear_name, x.name))


def filter_latest_table_info(table_info: Iterable[TableInfo]) -> Iterator[TableInfo]:
    for _, group in itertools.groupby(table_info, key=lambda x: x.clear_name):
        yield list(group)[-1]


def write_field(f: TextIO, field: bigquery.SchemaField):
    if field.field_type in ['TIME', 'TIMESTAMP', 'DATE', 'DATETIME']:
        f.write(f'  dimension_group: {field.name} {{\n')
    else:
        f.write(f'  dimension: {field.name} {{\n')
        if field.name == 'id':
            f.write('    primary_key: yes\n')

    if field.field_type == ['INTEGER', 'FLOAT', 'NUMERIC']:
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
    f.write(f'view: {info.clear_name} {{\n')
    f.write(f'  sql_table_name: `{info.project_id}.{info.dataset_id}.{info.clear_name}')
    if info.is_sharding:
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
        write_record_child(f, field, f'{info.clear_name}__')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_id')
    parser.add_argument('dataset_id')
    return parser.parse_args()


def main():
    args = parse_args()
    client = get_client()
    dataset = get_dataset(args.project_id, args.dataset_id)

    tables = get_tables(client, dataset)
    table_info = filter_latest_table_info(get_tables_info(tables))

    for info in table_info:
        info.create_dir()
        with info.path.open('w') as f:
            write_look_ml(f, info)


if __name__ == '__main__':
    main()
