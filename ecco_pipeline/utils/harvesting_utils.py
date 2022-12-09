from utils import solr_utils


def get_solr_docs(dataset_name: str):
    docs = {}
    descendants_docs = {}

    # Query for existing harvested docs
    fq = ['type_s:granule', f'dataset_s:{dataset_name}']
    harvested_docs = solr_utils.solr_query(fq)

    # Dictionary of existing harvested docs
    # harvested doc filename : solr entry for that doc
    if len(harvested_docs) > 0:
        for doc in harvested_docs:
            docs[doc['filename_s']] = doc

    # Query for existing descendants docs
    fq = ['type_s:descendants', f'dataset_s:{dataset_name}']
    existing_descendants_docs = solr_utils.solr_query(fq)

    # Dictionary of existing descendants docs
    # descendant doc date : solr entry for that doc
    if len(existing_descendants_docs) > 0:
        for doc in existing_descendants_docs:
            if 'hemisphere_s' in doc.keys() and doc['hemisphere_s']:
                key = (doc['date_s'], doc['hemisphere_s'])
            else:
                key = doc['date_s']
            descendants_docs[key] = doc

    return docs, descendants_docs


def check_update(docs, filename, mod_time):
    return (not filename in docs.keys()) or \
        (not docs[filename]['harvest_success_b']) or \
        (docs[filename]['download_time_dt'] < str(mod_time))


def make_ds_doc(config, source, chk_time):
    ds_meta = {}
    ds_meta['type_s'] = 'dataset'
    ds_meta['dataset_s'] = config['ds_name']
    ds_meta['short_name_s'] = config['original_dataset_short_name']
    ds_meta['source_s'] = source
    ds_meta['data_time_scale_s'] = config['data_time_scale']
    ds_meta['date_format_s'] = config['date_format']
    ds_meta['last_checked_dt'] = chk_time
    ds_meta['original_dataset_title_s'] = config['original_dataset_title']
    ds_meta['original_dataset_short_name_s'] = config['original_dataset_short_name']
    ds_meta['original_dataset_url_s'] = config['original_dataset_url']
    ds_meta['original_dataset_reference_s'] = config['original_dataset_reference']
    ds_meta['original_dataset_doi_s'] = config['original_dataset_doi']
    return ds_meta
