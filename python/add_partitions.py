import yaml
import copy
import json

with open('cluster-config-base.yaml', 'r') as f:
    conf = yaml.safe_load(f)

template = conf['Scheduling']['SlurmQueues'][0]

with open('partition-config.json') as f:
    partitions = json.load(f)
parts = list()

for pconfig in partitions:
    itype = pconfig['instance']
    name = itype.replace('.', '-')
    part = copy.deepcopy(template)
    part['ComputeResources'][0]['InstanceType'] = itype
    if 'efa' in pconfig and pconfig['efa']:
        part['ComputeResources'][0]['Efa']['Enabled'] = True
        part['ComputeResources'][0]['Networking']['PlacementGroup']['Enabled'] = True
    if 'maxcount' in pconfig:
        part['ComputeResources'][0]['MaxCount'] = pconfig['maxcount']
    if 'spot' in pconfig and pconfig['spot']:
        part['CapacityType'] = 'SPOT'
        name = "%s-spot" % name
    part['ComputeResources'][0]['Name'] = name
    part['Name'] = name

    parts.append(part)

conf['Scheduling']['SlurmQueues'] = parts

with open('cluster-config.yaml', 'w') as f:
        yaml.dump(conf, f)
