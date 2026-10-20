def usage(total=10, settlement='active', complete=False):
    return {'schema_version': 'ecologyrsi-dsh.session-usage/1',
        'identity': dict(run_id='r', stage='s', idempotency_key='k', child_reservation_id='c', session_id='session'),
        'session_metrics': {'schema_version': 'ecologyrsi-dsh.dsh-session-metrics/1', 'session_id': 'session',
            'context_pressure': {'available': False, 'source': 'dsh_token_meter'},
            'provider_usage': {'available': True, 'source': 'dsh_session_projection_token_usage',
                'measurement': 'cumulative_provider_reported_usage',
                'totals': dict(uncached_input_tokens=total, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, total_tokens=total)}},
        'settlement': settlement, 'usage_complete': complete}
