def get_configs(dataset, task):

    if dataset == 'rel-amazon' and task == 'user-churn':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 32,
        }
        loader_config = {
            'batch_size': 4096,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-amazon' and task == 'item-churn':
        model_config = {
            'num_model_layers': 4,
            'channels': 256,
            'aggr': 'sum',
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config      
    
    if dataset == 'rel-avito' and task == 'user-clicks':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-avito' and task == 'user-visits':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-event' and task == 'user-repeat':
        model_config = {
            'num_model_layers': 1,
            'channels': 32,
            'aggr': 'sum',
            'num_heads': 2,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-event' and task == 'user-ignore':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-f1' and task == 'driver-dnf':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 1,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-f1' and task == 'driver-top3':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 2,
            'simplified_MP': True,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-hm' and task == 'user-churn':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'user-engagement':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 1024,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'user-badge':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-trial' and task == 'study-outcome':
        model_config = {
            'num_model_layers': 2,
            'channels': 256,
            'aggr': 'sum',
            'num_heads': 1,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-amazon' and task == 'user-ltv':
        model_config = {
            'num_model_layers': 2,
            'channels': 32,
            'aggr': 'sum',
            'num_heads': 64,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-amazon' and task == 'item-ltv':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-avito' and task == 'ad-ctr':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-event' and task == 'user-attendance':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-f1' and task == 'driver-position':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config

    if dataset == 'rel-hm' and task == 'item-sales':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'post-votes':
        model_config = {
            'num_model_layers': 2,
            'channels': 64,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 1024,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-trial' and task == 'study-adverse':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'mean',
            'num_heads': 2,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 64,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-trial' and task == 'site-success':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'mean',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 64,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    
    if dataset == 'rel-amazon' and task == 'user-item-purchase':
        model_config = {
            'num_heads': 2,
        }
        loader_config = {
            'batch_size': 4096,
        }
        return model_config, loader_config

    if dataset == 'rel-amazon' and task == 'user-item-rate':
        model_config = {
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 512,
        }
        return model_config, loader_config

    if dataset == 'rel-amazon' and task == 'user-item-review':
        model_config = {
            'num_heads': 1,
        }
        loader_config = {
            'batch_size': 256,
        }
        return model_config, loader_config

    if dataset == 'rel-avito' and task == 'user-ad-visit':
        model_config = {
            'num_model_layers': 8,
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 256,
            'num_layers': 2,
        }
        return model_config, loader_config

    if dataset == 'rel-hm' and task == 'user-item-purchase':
        model_config = {
            'num_model_layers': 1,
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 2048,
            'num_layers': 2,
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'user-post-comment':
        model_config = {
            'num_model_layers': 4,
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_layers': 2,
        }
        return model_config, loader_config

    if dataset == 'rel-stack' and task == 'post-post-related':
        model_config = {
            'num_model_layers': 2,
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 512,
            'num_layers': 2,
        }
        return model_config, loader_config

    if dataset == 'rel-trial' and task == 'condition-sponsor-run':
        model_config = {
            'num_model_layers': 8,
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_layers': 4,
        }
        return model_config, loader_config
    
    if dataset == 'rel-trial' and task == 'site-sponsor-run':
        model_config = {
            'num_model_layers': 8,
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 64,
            'num_layers': 4,
        }
        return model_config, loader_config
