Location of our existing DAG repository of the existing DAG: 

`https://gitlab.com/fproj/data-team/airflow/airflow-pipelines/`

# Example: `airflow-pipelines/flows/dags/crawlers/vmware_esxi_firmware.py`
```
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.trigger_rule import TriggerRule

from flows.include.settings import UserName
from flows.include.settings.crawlers import vmware_crawler_settings

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=3)
}

with DAG(
    vmware_crawler_settings.DAG_NAME,
    default_args=default_args,
    description='VMWare ESXi web OS patches packages Crawler',
    schedule="0 0 * * 3",  # Every Wednesday, at 00:00
    start_date=datetime(2024, 7, 24, 8, 0, 0),
    catchup=False,  # No historic executions to do,
    tags=["vmware esxi crawler", "crawlers", "os patch"]
) as dag:
    crawler_kubernetes_operator = KubernetesPodOperator(
        **vmware_crawler_settings.CRAWLER_POD_SETTINGS,
        task_id=vmware_crawler_settings.DAG_NAME,
        env_vars={'KDB_API_IP': os.getenv("MONGO_KDB_IP")},
        run_as_user=UserName.NONE,
    )

    downloader_operator = KubernetesPodOperator(
        **vmware_crawler_settings.DOWNLOADER_POD_SETTINGS,
        cmds=['python', 'tools/download_files.py'],
        task_id='downloader_operator',
        env_vars={
            'GOOGLE_APPLICATION_CREDENTIALS': '/usr/.credentials/.google_keyfile.json',
            'KDB_API_IP': os.getenv('MONGO_KDB_IP')
        },
        trigger_rule=TriggerRule.ALL_DONE,
        run_as_user=UserName.NONE,
    )

    delete_xcoms_crawler_vmware_esxi_os_patch = TriggerDagRunOperator(
        task_id=f"delete_xcoms_{vmware_crawler_settings.DAG_NAME}",
        trigger_dag_id="delete_all_xcoms_dag",
        conf={"dag_id": vmware_crawler_settings.DAG_NAME},
    )

    (
        crawler_kubernetes_operator
        >> downloader_operator
        >> delete_xcoms_crawler_vmware_esxi_os_patch
    )
```

# Example: `airflow-pipelines/flows/dags/consumers/uefi_allowlister_consumer.py`

```
#!/usr/bin/env python3
# coding=utf-8

"""
UEFI Allowlister Consumer

1. Pull and acknowledge a single Pub/Sub message
2. Download the package UEFI images folder from GCS
3. Create the package EFI binaries folder in NFS
4. Run the UEFI Allowlister container pod
5. Upload the output EFI binary blobs to GCS
6. Insert and relate the EFI binary metadata to MongoDB
7. Update the UEFI Allowlisting status in MongoDB
8. Delete the working folder from NFS and DAG XComs
9. Re-trigger DAG in case more Pub/Sub messages exist
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

from flows.include.mongo.commands import update_uefi_allowlisting_status, upsert_uefi_allowlisting_metadata
from flows.include.settings import UserName
from flows.include.settings.connections import google_settings
from flows.include.settings.uefi_allowlister import uefi_allowlister_settings
from flows.include.utils.helpers import delete_xcoms, make_folder
from flows.plugins.operators.gcs import GcsFolderToLocalFolderOperator, LocalFolderToGcsFolderOperator
from flows.plugins.operators.kubernetes import BaseKubernetesPodOperator
from flows.plugins.operators.pubsub import PubSubToXcomOperator

with DAG(
    dag_id=uefi_allowlister_settings.DAG_ID_CONSUMER,
    dag_display_name=f'{uefi_allowlister_settings.PROJECT_NAME} Consumer',
    description=uefi_allowlister_settings.PROJECT_DESC,
    schedule=uefi_allowlister_settings.SCHEDULE,
    default_args=uefi_allowlister_settings.DEFAULT_ARGS,
    max_active_runs=uefi_allowlister_settings.DAG_MAX_ACTIVE_RUNS,
    max_active_tasks=uefi_allowlister_settings.DAG_MAX_ACTIVE_TASKS,
    tags=['UEFI Post-Processing', uefi_allowlister_settings.PROJECT_NAME, 'Consumer']
) as uefi_allowlister_consumer_dag:
    pull_pubsub_msg: PubSubToXcomOperator = PubSubToXcomOperator(
        task_id='pull_pubsub_msg',  # Do not change task_id (needed for downstream xcoms)
        task_display_name='Pull Pub/Sub Message',
        doc='Pull and acknowledge a single Pub/Sub message',
        gcp_conn_id=google_settings.CONN_ID,
        project_id=uefi_allowlister_settings.GOOGLE_PROJECT_ID,
        subscription=uefi_allowlister_settings.SUBSCRIPTION,
        max_messages=uefi_allowlister_settings.CONSUMER_MSG_LEN,
        ack_messages=uefi_allowlister_settings.CONSUMER_MSG_ACK,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    gcs_download_folder: GcsFolderToLocalFolderOperator = GcsFolderToLocalFolderOperator(
        task_id='gcs_download_folder',
        task_display_name='Download UEFI images from GCS',
        doc='Download the package UEFI images folder from GCS',
        bucket_name=google_settings.CRAWLER_BUCKET,
        gcs_folder=uefi_allowlister_settings.UNPACKING_PATH,
        local_folder=uefi_allowlister_settings.INPUT_FOLDER_PATH,
        retries=uefi_allowlister_settings.RETRY_COUNT_GCS,
        retry_delay=uefi_allowlister_settings.RETRY_DELAY_GCS,
        retry_exponential_backoff=uefi_allowlister_settings.RETRY_BACKOFF_GCS,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    gcs_upload_folder: LocalFolderToGcsFolderOperator = LocalFolderToGcsFolderOperator(
        task_id='gcs_upload_folder',
        task_display_name='Upload EFI binaries to GCS',
        doc='Upload the output EFI binary blobs to GCS',
        run_as_user=UserName.ROOT,
        bucket_name=google_settings.UEFI_BINARIES_BUCKET,
        gcs_folder=uefi_allowlister_settings.OUTPUT_GCS_PATH,
        folder_path=uefi_allowlister_settings.OUTPUT_RESULT_PATH,
        retries=uefi_allowlister_settings.RETRY_COUNT_GCS,
        retry_delay=uefi_allowlister_settings.RETRY_DELAY_GCS,
        retry_exponential_backoff=uefi_allowlister_settings.RETRY_BACKOFF_GCS,
        overwrite=uefi_allowlister_settings.GCS_UPLOAD_DIR_OVR,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    uefi_allowlister: BaseKubernetesPodOperator = BaseKubernetesPodOperator(
        task_id=f'run_{uefi_allowlister_settings.PROJECT_ID}',
        task_display_name=f'Run {uefi_allowlister_settings.PROJECT_NAME}',
        doc=f'Run the {uefi_allowlister_settings.PROJECT_NAME} container pod',
        name=f'run_{uefi_allowlister_settings.PROJECT_ID}',
        image=f'{uefi_allowlister_settings.REPOSITORY}:{uefi_allowlister_settings.TAG}',
        arguments=uefi_allowlister_settings.ARGUMENTS_POD,
        retries=uefi_allowlister_settings.RETRY_COUNT_POD,
        retry_delay=uefi_allowlister_settings.RETRY_DELAY_POD,
        retry_exponential_backoff=uefi_allowlister_settings.RETRY_BACKOFF_POD,
        execution_timeout=uefi_allowlister_settings.TIMEOUT_POD,
        node_selector=uefi_allowlister_settings.NODE_POOL_POD,
        cpu_request=uefi_allowlister_settings.CPU_REQUEST_POD,
        cpu_limit=uefi_allowlister_settings.CPU_LIMIT_POD,
        memory_request=uefi_allowlister_settings.MEMORY_REQUEST_POD,
        memory_limit=uefi_allowlister_settings.MEMORY_LIMIT_POD,
        mount_empty_dir=uefi_allowlister_settings.EMPTY_DIR_POD,
        affinity_one_pod_per_node=uefi_allowlister_settings.SOLE_AFFINITY_POD,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    create_output_folder: PythonOperator = PythonOperator(
        task_id='create_output_folder',
        task_display_name='Create output folder',
        doc='Create the package output folder in NFS',
        python_callable=make_folder,
        op_kwargs=uefi_allowlister_settings.KWARGS_CREATE_OUTPUT_FOLDER,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    upsert_metadata: PythonOperator = PythonOperator(
        task_id='upsert_metadata',
        task_display_name='Upsert EFI metadata to MongoDB',
        doc='Upsert and relate the EFI binary metadata to MongoDB',
        run_as_user=UserName.ROOT,
        python_callable=upsert_uefi_allowlisting_metadata,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    set_mongo_status: PythonOperator = PythonOperator(
        task_id='set_mongo_status',
        task_display_name='Set status to MongoDB',
        doc='Set the UEFI Allowlisting status in MongoDB',
        python_callable=update_uefi_allowlisting_status,
        trigger_rule=TriggerRule.NONE_SKIPPED
    )

    cleanup_paths: BashOperator = BashOperator(
        task_id='cleanup_paths',
        task_display_name='Cleanup working paths',
        doc='Delete the package folder from NFS',
        run_as_user=UserName.ROOT,
        bash_command=uefi_allowlister_settings.BASH_CLEANUP_PATHS,
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    cleanup_xcoms: PythonOperator = PythonOperator(
        task_id='cleanup_xcoms',
        task_display_name='Cleanup XCom messages',
        doc='Delete the DAG Run XComs',
        python_callable=delete_xcoms,
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    self_trigger: TriggerDagRunOperator = TriggerDagRunOperator(
        task_id='trigger_consumer_dag',
        task_display_name='Trigger Consumer DAG',
        doc='Re-trigger DAG in case more Pub/Sub messages exist',
        trigger_dag_id=uefi_allowlister_settings.DAG_ID_CONSUMER,
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    # pylint: disable=pointless-statement
    pull_pubsub_msg >> gcs_download_folder >> create_output_folder >> uefi_allowlister >> gcs_upload_folder >> \
    upsert_metadata >> set_mongo_status >> cleanup_paths >> self_trigger >> cleanup_xcoms

```

# Example: `airflow-pipelines/flows/dags/producers/uefi_producer_generator.py`
```
#!/usr/bin/env python3
# coding=utf-8

"""
UEFI Producer DAG Generator

1. Clear Pub/Sub subscription queue
2. Run the Bucket Normalization DAG
3. Delete the project folder from NFS
4. Store MongoDB query results to CSV
5. Upload MongoDB results CSV to GCS
6. Publish each CSV row to Pub/Sub
7. Trigger project Consumer DAGs
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.api.client.local_client import Client
from airflow.decorators import task
from airflow.models.dagrun import DagRun
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils import timezone
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.types import DagRunType

from flows.include.settings import UserName
from flows.include.settings.connections import google_settings
from flows.include.settings.uefi_unpacker import uefi_unpacker_settings
from flows.include.utils.helpers import delete_xcoms
from flows.plugins.operators.gcs import LocalFileToGcsObjectOperator
from flows.plugins.operators.mongo import MongoAggregationToCSVOperator
from flows.plugins.operators.pubsub import CsvToPubSubOperator, PubSubPurgeQueueOperator


def generate_uefi_producer_dag(
    project_tags: list[str],
    dag_id_consumer: str,
    dag_id_producer: str,
    schedule: str | timedelta | None,
    mongo_collection: str,
    mongo_pipeline: list[dict[str, dict]],
    csv_filepath: str,
    csv_fields: list[str],
    gcs_object_name: str,
    project_folder: str,
    project_name: str,
    pubsub_topic: str,
    pubsub_subscription: str,
    default_args: dict[str, Any] | None,
    google_project_id: str = 'dataproject-x',
    dag_run_ratio: int = 1,
    max_dag_runs: int = 132
) -> DAG:
    """
    :param project_tags: List of project-related DAG tags
    :param dag_id_consumer: The Consumer DAG ID
    :param dag_id_producer: The Producer DAG ID
    :param schedule: The DAG schedule interval
    :param mongo_collection: The MongoDB collection to run the aggregation pipeline
    :param mongo_pipeline: The aggregation query pipeline to run in MongoDB
    :param csv_filepath: The file path of the CSV file to create and upload to GCS
    :param csv_fields: The list of fields to write in the CSV header
    :param gcs_object_name: The object name to set when uploading the CSV file to GCS
    :param default_args: Set different default_args for the new DAG
    :param project_folder: The project root directory in NFS
    :param project_name: The name of the project
    :param pubsub_topic: The Google PubSub topic to send each CSV row as a message
    :param pubsub_subscription: The Google PubSub subscription where the consumers pull messages from
    :param google_project_id: The Google project ID to use in PubSub.
    :param dag_run_ratio: ratio for triggering consumers DAGs depending on number of messages in PubSub queue.
        Example: when dag_run_ratio=2, 1 consumer DAG is triggered every 2 messages in the PubSub queue.
    :param max_dag_runs: Maximum number of consumer DAGs that can be triggered and can run at the same time.

    :return: Producer DAG
    """

    with DAG(
        dag_id=dag_id_producer,
        dag_display_name=f'{project_name} Producer',
        description=f'Produce {project_name} Consumers',
        schedule=schedule,
        default_args=default_args,
        tags=[*project_tags, 'Producer'],
        params={
            'dag_run_ratio': Param(
                default=dag_run_ratio,
                type='integer',
                title='DAG run ratio',
                description='E.g. value 2 means that 1 Consumer DAG is triggered every 2 Pub/Sub messages'),
            'max_dag_runs': Param(
                default=max_dag_runs,
                type='integer',
                title='Max DAG runs',
                description='Maximum number of Consumer DAGs to run concurrently',
            )
        }
    ) as uefi_producer_dag:
        purge_pubsub_queue: PubSubPurgeQueueOperator = PubSubPurgeQueueOperator(
            task_id='purge_pubsub_queue',
            task_display_name='Purge Pub/Sub Queue',
            doc='Clear the Pub/Sub subscription queue',
            gcp_conn_id=google_settings.CONN_ID,
            project_id=google_project_id,
            subscription=pubsub_subscription,
            trigger_rule=TriggerRule.ALL_SUCCESS
        )

        if dag_id_producer == uefi_unpacker_settings.DAG_ID_PRODUCER:
            normalize_bucket: TriggerDagRunOperator = TriggerDagRunOperator(
                task_id='run_bucket_normalizer',
                task_display_name='Run Bucket Normalizer',
                doc='Run the Bucket Normalization DAG',
                trigger_dag_id='bucket_normalization_dag',
                wait_for_completion=True,
                reset_dag_run=True,
                trigger_rule=TriggerRule.ALL_SUCCESS
            )
        else:
            normalize_bucket: EmptyOperator = EmptyOperator(
                task_id='skip_bucket_normalizer',
                task_display_name='Skip Bucket Normalizer',
                doc='Skip the Bucket Normalization DAG',
                trigger_rule=TriggerRule.ALL_SUCCESS
            )

        cleanup_paths: BashOperator = BashOperator(
            task_id='clean_project_folder',
            task_display_name=f'Cleanup {project_name} path',
            doc=f'Delete the {project_name} folder from NFS',
            run_as_user=UserName.ROOT,
            bash_command=f'rm -fdR "{project_folder}"',
            trigger_rule=TriggerRule.ALL_SUCCESS
        )

        query_mongo: MongoAggregationToCSVOperator = MongoAggregationToCSVOperator(
            task_id='query_mongo',
            task_display_name='Query MongoDB pipeline',
            doc='Store MongoDB query results to CSV',
            mongo_collection=mongo_collection,
            mongo_pipeline=mongo_pipeline,
            csv_filepath=csv_filepath,
            csv_fields=csv_fields,
            trigger_rule=TriggerRule.ALL_SUCCESS
        )

        upload_to_gcs: LocalFileToGcsObjectOperator = LocalFileToGcsObjectOperator(
            task_id='upload_to_gcs',
            task_display_name='Upload MongoDB query to GCS',
            doc='Upload MongoDB results CSV to GCS',
            bucket_name=google_settings.FLOWS_BUCKET,
            object_name=gcs_object_name,
            file_path=csv_filepath,
            trigger_rule=TriggerRule.ALL_SUCCESS
        )

        publish_to_pubsub: CsvToPubSubOperator = CsvToPubSubOperator(
            task_id='publish_to_pubsub',
            task_display_name='Publish messages to Pub/Sub',
            doc='Publish each CSV row to Pub/Sub',
            gcp_conn_id=google_settings.CONN_ID,
            topic=pubsub_topic,
            project_id=google_project_id,
            csv_filepath=csv_filepath,
            trigger_rule=TriggerRule.ALL_SUCCESS
        )

        cleanup_xcoms: PythonOperator = PythonOperator(
            task_id='cleanup_xcoms',
            task_display_name='Cleanup XCom messages',
            doc='Delete the DAG Run XComs',
            python_callable=delete_xcoms,
            trigger_rule=TriggerRule.ONE_SUCCESS
        )

        @task(
            task_id='trigger_consumer_dag',
            task_display_name=f'Trigger {project_name} Consumer',
            doc=f'Trigger {project_name} Consumer DAGs',
            retries=0,
            trigger_rule=TriggerRule.ALL_SUCCESS
        )
        def trigger_consumer_dag(**context) -> None:
            """
            Triggers the consumer DAG based on the total number of messages published to Pub/Sub.

            It calculates the number of consumer DAG runs required by dividing the total messages
            by the `dag_run_ratio` and limiting it to `max_dag_runs`, if necessary.
            """

            messages_count: int = context['ti'].xcom_pull(
                task_ids=publish_to_pubsub.task_id,
                key=publish_to_pubsub.TOTAL_MSGS_XCOM_KEY
            )

            if not messages_count:
                logging.warning('No messages have been published to Pub/Sub, nothing to trigger!')

                return

            dag_runs_consumer: int = min(
                context['params']['max_dag_runs'],
                max(messages_count // context['params']['dag_run_ratio'], 1)
            )

            logging.info('Triggering %s %s Consumers...', dag_runs_consumer, project_name)

            api_local_client: Client = Client(api_base_url=None, auth=None, session=None)

            for _ in range(dag_runs_consumer):
                datetime_consumer: datetime = timezone.utcnow()

                run_id_consumer: str = DagRun.generate_run_id(
                    run_type=DagRunType.MANUAL,
                    execution_date=datetime_consumer
                )

                dag_run_trigger: dict | None = api_local_client.trigger_dag(
                    dag_id=dag_id_consumer,
                    run_id=run_id_consumer,
                    execution_date=datetime_consumer,
                    replace_microseconds=False
                )

                if isinstance(dag_run_trigger, dict):
                    logging.info('Triggered %s DAG ID %s with Run ID %s',
                                 project_name, dag_id_consumer, run_id_consumer)

                    logging.debug(json.dumps(dag_run_trigger, sort_keys=True, indent=4, default=str))

        # pylint: disable=pointless-statement,expression-not-assigned
        purge_pubsub_queue >> normalize_bucket >> cleanup_paths >> query_mongo >> upload_to_gcs >> \
        publish_to_pubsub >> trigger_consumer_dag() >> cleanup_xcoms

    return uefi_producer_dag
```