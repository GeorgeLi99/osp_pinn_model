"""
Copyright (c) 2024 ANSYS, Inc.

This program is commercial software: you can use it under the terms of the
Ansys License Agreement as published by ANSYS, Inc.

Except as expressly permitted in the Ansys License Agreement, you may not
modify or redistribute this program.
"""

from subprocess import check_output, STDOUT, CalledProcessError
import json
from os import listdir
from os.path import join
from os.path import isfile, expanduser, join, split, splitext, dirname
from base64 import b64encode
from time import sleep

import sys
import gzip
import time
import datetime
import logging
import string
import random

# Depending on Python 2.x vs. Python 3 this is a bit different
try:
    import StringIO
except ImportError:
    from io import StringIO

# generate key for no-password communication between compute node
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

MAJOR_RELEASE  = 'v242'
MODERN_LUMLDIR = "/opt/lumerical/" + MAJOR_RELEASE

FDTD_ROOT   = MODERN_LUMLDIR
MODE_ROOT   = MODERN_LUMLDIR
DEVICE_ROOT = MODERN_LUMLDIR

FDTD_BIN   = FDTD_ROOT + '/bin'
MODE_BIN   = MODE_ROOT + '/bin'
DEVICE_BIN = DEVICE_ROOT + '/bin'

FDTD_CAD   = FDTD_BIN + '/fdtd-solutions'
MODE_CAD   = MODE_BIN + '/mode-solutions'
DAVICE_CAD = DEVICE_BIN + '/device'

FDTD_INI   = MODERN_LUMLDIR + r'/Lumerical/FDTD\ Solutions.ini'
MODE_INI   = MODERN_LUMLDIR + r'/Lumerical/MODE\ Solutions.ini'
DEVICE_INI = MODERN_LUMLDIR + r'/Lumerical/Lumerical\ DEVICE.ini'


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.interval > 10:  # do not print if under 10 seconds
            value = datetime.datetime.fromtimestamp(86400 + self.interval)  # Python 3: add an day but do not show it.
            print(value.strftime('Timer: %M minutes and %S seconds'))


class VpcConfigurationError(Exception):
    """Exception class to indicate an error in the configuration of the VPC"""
    pass


def configure_logging(log_level=logging.WARNING):
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)


def _shell_command(command, error_message=None, raise_exception=True):
    """
    Helper function to make system calls. The command string passed will be run as if typed into a Terminal or Command Prompt.

    :param command: Command string to be run
    :param error_message: Pass an error message that will be displayed to the user
    :param raise_exception: True or False. If True any exception will be raised. If False the program will exit.

    :return: shell_output: String
    """
    logger = logging.getLogger(__name__)

    try:
        logger.debug('Running Command:    {0}'.format(command))
        with Timer() as _:
            shell_output = check_output(command, stderr=STDOUT, universal_newlines=True, shell=True)
        return shell_output

    except CalledProcessError as exc:
        if error_message: logger.error(error_message)
        if raise_exception:
            raise
        else:
            print(exc.output)
            sys.exit()


def _parse_json_shell_output(shell_output, variable_name=None):
    """
    Helper function to decode shell output, parse the JSON string and extract a single entry from the top-level dictionary.

    :param shell_output: Raw text containing JSON data to be parsed
    :param variable_name: Name of a variable in the JSON data to be extracted

    :return: JSON data of the requested variable
    """

    output_json = json.loads(shell_output)
    if variable_name:
        return output_json[variable_name]
    else:
        return output_json


def _check_aws_account():
    """ Checks that valid credentials for an AWS account have been entered into the CLI. """
    logger = logging.getLogger(__name__)

    command = 'aws ec2 describe-account-attributes --output json --attribute-name supported-platforms' \
              ' --query AccountAttributes'
    error_msg = 'The AWS CLI could not be accessed, please ensure that the AWS CLI is correctly installed.'
    attribute_values = _shell_command(command, error_msg)
    if 'VPC' not in attribute_values:
        logger.error('The AWS CLI account is not properly configured or does not support the VPC platform.')
        raise VpcConfigurationError('The AWS CLI account is not properly configured or does not support the VPC platform.')


def _setup_json_file_name(file_name='setupLumericalVPC.json', current_dir=dirname(__file__), prefix='file://'):
    """
    Looks for the JSON setup file inside the current directory and its subdirectories.

    :param pattern: Name of the file containing the CloudFormation template in JSON format
    :param current_dir: Current directory

    :return: the name string required by calls to aws cloudformation --stack-name.
    """
    if file_name in listdir(current_dir):
        return prefix + join(current_dir, file_name)
    else:
        raise RuntimeError("Cannot find {} in {}".format(file_name, current_dir))


def _is_name_valid(name):
    return len(name) > 0 and name[0].isalnum() and name.replace('-', '').isalnum()


def _get_stack_name(name):
    """
    Appends a suffix to the user provided name to tag the stack
    :param name: A generic name tag used for all resources related to the CloudFormation stack

    :return: the name string plus '-stack' appended
    """
    return name + '-stack'


def _get_vpc_name(name):
    """
    Appends a suffix to the user provided name to tag the VPC

    :param name: A generic name tag used for all resources related to the CloudFormation stack

    :return: the name string plus '-vpc' appended
    """
    return name + '-vpc'


def _json_list_to_dict(lst, key, value):
    """
    Converts a JSON list with individual entries for keys and values into a proper dictionary
    :param lst: The list of strings
    :param key: The string tagging a key (e.g. 'ParameterKey')
    :param value: The string tagging a value (e.g. 'ParameterValue')

    :return: A proper Python dictionary generated from the list entries
    """
    result = dict()

    for cur_entry in lst:
        result[cur_entry[key]]=cur_entry[value]
    return result


def _create_stack(name, own_ip):
    """
    Uses a call to AWS CloudFormation to create a new stack with a single VPC. A stack can contain
    multiple VPCs, however, we'll assume for now that the stack will only contain a single VPC. The
    configuration of the stack is in a separate JSON file returned by _setup_json_file_name()

    :param name: Generic name tag given to all resources related to the stack.
    """
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)
    vpc_name = _get_vpc_name(name)

    json_template_file_name = _setup_json_file_name()
    if not json_template_file_name:
        raise VpcConfigurationError("Could not find the JSON template file called 'setupLumericalVPC.json'. Please "
                                    "make sure that this file is present in the current directory or in a "
                                    "subdirectory.")

    parameters = 'ParameterKey=EnvironmentName,ParameterValue={0} ParameterKey=Port22AccessCIDR,ParameterValue={1}'.format(vpc_name,own_ip)
    command = 'aws cloudformation create-stack --stack-name {0} --template-body "{1}" --capabilities CAPABILITY_NAMED_IAM --parameters {2}'.format(stack_name, json_template_file_name, parameters)
    try:
        _shell_command(command, raise_exception=True)

        # Wait for confirmation that VPC has been created.
        command = 'aws cloudformation wait stack-create-complete --stack-name {0}'.format(stack_name)
        error_msg = 'The requested VPC could not be successfully created.'
        _shell_command(command, error_msg)
    except CalledProcessError as exc:
        # If the error is that the stack already exists, then we ignore it, otherwise, we just pass along the error.
        if "already exists" not in exc.output:
            raise RuntimeError(exc.output)


def _get_stack_info(stack_name):
    # Retrieve a full report with the stack information.
    command = 'aws cloudformation describe-stacks --output json --stack-name {0} --query "Stacks[0]"'.format(stack_name)
    shell_output = _shell_command(command)
    stack_id = _parse_json_shell_output(shell_output, 'StackId')
    description = _parse_json_shell_output(shell_output, 'Description')
    parameters = _parse_json_shell_output(shell_output, 'Parameters')
    outputs = _parse_json_shell_output(shell_output, 'Outputs')

    # Convert list of parameters to a dictionary
    parameters = _json_list_to_dict(parameters, 'ParameterKey', 'ParameterValue')
    subnet_cidr = parameters['PrivateSubnetCIDR']
    ami_id = parameters['LumericalAmiId']
    lic_ami_id = parameters['LicenseAmiId']

    # Convert list of outputs to a dictionary
    outputs = _json_list_to_dict(outputs, 'OutputKey', 'OutputValue')
    vpc_id = outputs["VPCID"]
    account_no = outputs["AccountNo"]
    availability_zone = outputs["Zone"]
    subnet_id = outputs["SubnetId"]
    security_grp_id = outputs["SecurityGroupId"]
    return description, stack_id, vpc_id, subnet_cidr, lic_ami_id, ami_id, account_no, availability_zone, subnet_id, security_grp_id, parameters


def _describe_vpc(name):
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)
    vpc_name = _get_vpc_name(name)

    description, stack_id, vpc_id, subnet_cidr, lic_ami_id, ami_id, account_no, availability_zone, subnet_id, security_grp_id, parameters = _get_stack_info(stack_name)

    logger.debug(description)
    logger.debug('Stack ID: {0}'.format(stack_id))
    logger.debug('VPC ID: {0}'.format(vpc_id))
    logger.debug('Subnet CIDR: {0}'.format(subnet_cidr))
    logger.debug('License AMI ID: {0}'.format(lic_ami_id))
    logger.debug('AMI ID: {0}'.format(ami_id))
    logger.debug('Subnet ID: {0}'.format(subnet_id))
    logger.debug('Security Group ID: {0}'.format(security_grp_id))

    # List all network interfaces and filter for the one associated with the VPC
    command = 'aws ec2 describe-network-interfaces --output json --filters Name=tag-value,Values="{0}-ENI"'.format(vpc_name)
    shell_output = _shell_command(command)
    net_interfaces = _parse_json_shell_output(shell_output, "NetworkInterfaces")

    # Make sure that we have exactly one matching network interface
    if len(net_interfaces) != 1:
        logger.error("There should be one network interface associated with a VPC. Found {}.".format(len(net_interfaces)))
        raise VpcConfigurationError("There should be one network interface associated with a VPC. Found {}.".format(len(net_interfaces)))
    mac_address = net_interfaces[0]["MacAddress"]

    # Print the required licensing information.
    output_string = ("\nThe following information is required to obtain your Lumerical cloud license:\n"
                     "  AWS account number: {0}\n"
                     "  AWS availability zone: {1}\n"
                     "  MAC address of license server network adapter: {2}\n"
                     "Please contact Lumerical sales with this information to obtain the required license.")

    print(output_string.format(account_no, availability_zone, mac_address))

    return vpc_id


def create_virtual_private_cloud(name, own_ip = "0.0.0.0/0"):
    """
    Creates a virtual private cloud (VPC) with an elastic IP, gateway, route, subnet, network interface and simple storage
    service (S3) endpoint. It reports the Amazon web services (AWS) account number, availability zone and MAC address of the
    lincese server network adapter associated with the newly created VPC. This information is required to obtain a license.

    :param name:   Name tag to identify the VPC and its associated resources.
    :param own_ip: IP range in CIDR notation used to restrict inbound network traffic.

    :return: ID of the VPC.

    Note: a VPC only needs to be created once before launching instances with the Lumerical AMI.
    """
    logger = logging.getLogger(__name__)

    _check_aws_account()

    if not _is_name_valid(name):
        logger.error('Invalid VPC name: it must start with an alphanumeric character and contain only alphanumeric characters or dashes.')
        return
    else:
        stack_name = _get_stack_name(name)
        vpc_name = _get_vpc_name(name)
        command = 'aws ec2 describe-vpcs --filter "Name=tag-value,Values={0},{1}" --query "Vpcs[0]"'.format(stack_name, vpc_name)
        shell_output = _shell_command(command)
        if stack_name in shell_output or vpc_name in shell_output:
            raise VpcConfigurationError('A VPC with the name "{0}" has already been created.'.format(name))
        else:
            # Create a new stack and wait for creation to complete
            print('Please wait while we create a new virtual private cloud. This operation can take several minutes.')
            _create_stack(name,own_ip)
        # Print the account and network adapter information.
        vpc_id = _describe_vpc(name)
        return vpc_id


def _delete_and_create_gateway(vpc_id, region, service_type, subnet_id, security_grp_id):
    """
    Deletes an existing interface gateway of specified type and recreates it. This is used for repairing our VPCs.
    """
    _delete_endpoints(vpc_id, region, service_type, subnet_id, security_grp_id)

    command = 'aws ec2 create-vpc-endpoint --vpc-endpoint-type Interface --vpc-id {0} --service-name com.amazonaws.{1}.{2} --subnet-ids {3} --security-group-ids {4}'.format(vpc_id, region, service_type, subnet_id, security_grp_id)
    _shell_command(command)


def _delete_endpoints(vpc_id, region, service_type, subnet_id, security_grp_id):
    """
    Deletes an existing interface gateway of specified type
    """
    command = 'aws ec2 describe-vpc-endpoints --output json --filters Name=vpc-id,Values="{0}" Name=service-name,Values="com.amazonaws.{1}.{2}"'.format(vpc_id, region, service_type)
    shell_output = _shell_command(command)
    list_of_endpoints = _parse_json_shell_output(shell_output, 'VpcEndpoints')

    if len(list_of_endpoints) > 0:
        command = 'aws ec2 delete-vpc-endpoints --vpc-endpoint-ids {0}'.format(list_of_endpoints[0]["VpcEndpointId"])
        _shell_command(command)


def _add_ssm_interface_gateways(stack_name):
    """
    Creates or re-creates two interface endpoints from the VPC to SSM (Simple Systems Manager) and to EC2MESSAGES to allow
    for remotely managing instances in the VPC through SSM. Ideally, this would be created using CloudFormation but currently
    CloudFormation does not support the creation of interface endpoints (only gateway endpoints).

    :param stack_name: Name of the CloudFormation stack which contains the VPC and the other networking components
    """
    description, stack_id, vpc_id, subnet_cidr, lic_ami_id, ami_id, account_no, region, subnet_id, security_grp_id, parameters=_get_stack_info(stack_name)

    _delete_and_create_gateway(vpc_id, region, "ssm", subnet_id, security_grp_id)
    _delete_and_create_gateway(vpc_id, region, "ec2messages", subnet_id, security_grp_id)


def set_individual_parameter(param_name, param_value):
    if param_value is not None:
        return 'ParameterKey={0},ParameterValue={1}'.format(param_name,param_value)
    else:
        return 'ParameterKey={0},UsePreviousValue=true'.format(param_name)


def _construct_parameters(name=None, license_file_content=None, key_pair_name=None, ami_id=None, lic_ami_id=None,
                          reset_tag=None, setup_compute_nodes=None, compute_node_instance_type=None,
                          num_compute_nodes=None, compute_node_max_storage=None):
    params = list()
    params.append(set_individual_parameter('EnvironmentName', name))
    params.append(set_individual_parameter('LicenseFile', license_file_content))
    params.append(set_individual_parameter('KeyPairName', key_pair_name))
    params.append(set_individual_parameter('LumericalAmiId', ami_id))
    params.append(set_individual_parameter('LicenseAmiId', lic_ami_id))
    params.append(set_individual_parameter('ResetTag', reset_tag))
    params.append(set_individual_parameter('SetupComputeNodes', setup_compute_nodes))
    params.append(set_individual_parameter('ComputeGroupCapacity', num_compute_nodes))
    params.append(set_individual_parameter('ComputeNodeDiskSize', compute_node_max_storage))
    params.append(set_individual_parameter('ComputeNodeInstanceType', compute_node_instance_type))

    return ' '.join(params)


def initialize_virtual_private_cloud(name, license_file, ami_id, lic_ami_id, key_pair_name=None):
    """
    Updates an existing virtual private cloud (VPC) with the provided license file, key pair and Amazon machine
    image (AMI) identifiers. A license server with the provided license file is launched automatically. If the
    VPC contains a running license server, it must be stopped beforehand.

    :param name:          Base name used to generate the VPC and its associated resources.
    :param license_file:  License file provided by Lumerical.
    :param key_pair_name: Name of a key pair used to gain SSH access to all VPC instances.
    :param ami_id:        AMI to be used for creating the compute nodes.
    :param lic_ami_id:    AMI to be used for creating and launching the license server.
    """
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)
    vpc_name = _get_vpc_name(name)
    command = 'aws ec2 describe-vpcs --filter "Name=tag-value,Values={0},{1}" --query Vpcs[0]'.format(vpc_name, stack_name)
    shell_output = _shell_command(command)
    if vpc_name not in shell_output and stack_name not in shell_output:
        logger.error("A VPC with the name '{0}' has not yet been created. Please use create_virtual_private_cloud() to create one.".format(name))
        return

    lic_server_id, lic_server_status = _find_license_server(name, False)
    if lic_server_status in [0, 16, 32, 64]:
        logger.error("Can't re-initialize the VPC while the license server status is '{0}'."
                     " Please call stop_license_server() first."
                     .format(_instance_status_code_to_string(lic_server_status)))
        return

    if not _is_name_valid(lic_ami_id):
        logger.error('The provided license server AMI ID is invalid.')
        return
    command = 'aws ec2 describe-images --image-ids {0} --query Images[*].[State]'.format(lic_ami_id)
    shell_output = _shell_command(command)
    if 'available' not in shell_output:
        logger.error('The provided license server AMI (ID:{0}) could not be found.'.format(lic_ami_id))
        return

    if not _is_name_valid(ami_id):
        logger.error('The provided compute instance AMI ID is invalid.')
        return
    command = 'aws ec2 describe-images --image-ids {0} --query Images[*].[State]'.format(ami_id)
    shell_output = _shell_command(command)
    if 'available' not in shell_output:
        logger.error('The provided compute instance AMI (ID:{0}) could not be found.'.format(ami_id))
        return

    # If a license file was provided, we read the content and pass it on as a parameter
    license_file = expanduser(license_file)
    if not isfile(license_file):
        logger.error('A license file with the name "{0}" could not be found.'.format(license_file))
        return
    with open(license_file, 'rb') as lic_file:
        license_file_content = b64encode(_gzip_data(lic_file.read())).decode('ascii')

    print('Please wait while we initialize your virtual private cloud. This operation can take several minutes.')

    print('Cleaning up existing VPC...')
    params = _construct_parameters(vpc_name, license_file_content, key_pair_name,
                                   ami_id, lic_ami_id, num_compute_nodes=0, reset_tag='true')
    _update_stack(name, params)

    print('Creating new VPC configuration...')
    params = _construct_parameters(vpc_name, license_file_content, key_pair_name,
                                   ami_id, lic_ami_id, num_compute_nodes=0, reset_tag='false')
    _update_stack(name, params)

    print('Adding interface gateways...')
    stack_name = _get_stack_name(name)
    _add_ssm_interface_gateways(stack_name)

    _report_initialized(name)


def _report_initialized(name):
    stack_name = _get_stack_name(name)
    _, _, _, _, _, _, _, _, _, _, parameters = _get_stack_info(stack_name)

    lic_server_id, lic_server_status = _find_license_server(name)
    if lic_server_id is None:
        raise RuntimeError('No license server ID could be found.')

    # the server is started automatically by cloudformation
    print('Starting license server...')
    command = 'aws ec2 wait instance-status-ok --instance-ids {0}'.format(lic_server_id)
    _shell_command(command)

    if _license_server_ready(lic_server_id, lic_server_status):
        stack_name = _get_stack_name(name)
        description, stack_id, vpc_id, subnet_cidr, lic_ami_id, ami_id, account_no, availability_zone, subnet_id, security_grp_id, parameters = _get_stack_info(stack_name)

        print('')
        print('Created and started license server:')
        print("    AMI: {}".format(parameters["LicenseAmiId"]))
        print("    Instance Type: t2.micro") # always will be micro, would be  nice to query this though
        if "SetupComputeNodes" in parameters and parameters["SetupComputeNodes"] == "true":
            print('')  # newline
            print("Compute instances initialized to defaults:")
            print("    AMI: {}".format(parameters["LumericalAmiId"]))
            print("    Instance Type: {}".format(parameters["ComputeNodeInstanceType"]))
            print("    Number of instances: {}".format(parameters["ComputeGroupCapacity"]))
        print('')
        print('The virtual private cloud initialization is now complete and the license server is running.')
        print('If you do not plan to run any simulations, please use the stop_license_server() function to stop the license server.')
        print('To query the status of the license manager at any time, please use the license_server_status() function.')
    else:
        pass  # the warnings are printed in _license_server_ready()


def _update_stack(name, params):
    """
    Update the CloudFormation stack to configure the license server
    """
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)
    command = 'aws cloudformation update-stack --stack-name {0} --template-body "{1}"  --capabilities ' \
              'CAPABILITY_NAMED_IAM --parameters {2}'.format(stack_name, _setup_json_file_name(), params)

    stack_is_updated = True
    try:
        _shell_command(command, raise_exception=True)
    except CalledProcessError as exc:
        # If the error is just that there is nothing to change, we keep going, otherwise, we exit.
        if "No updates are to be performed" not in exc.output:
            logger.error(exc.output)
            raise RuntimeError(exc.output)
        else:
            # If the error just is "No updates are to be performed", we ignore it.
            logger.debug("No updates are to be performed.")
            stack_is_updated = False

    # Wait until the update is complete.
    if stack_is_updated:
        command = 'aws cloudformation wait stack-update-complete --stack-name {0}'.format(stack_name)
        error_msg = 'The requested virtual private cloud could not be successfully updated.'
        _shell_command(command, error_msg)


def _find_license_server(name, log_error=True):
    """
    Searches for active (either running or pending) instances of a License server in the VPC with the provided name.

    :param name: Name used by the create_virtual_private_cloud() function to create a VPC.
    """

    vpc_name = _get_vpc_name(name)

    command = 'aws ec2 describe-instances --output json --filters Name=tag-value,Values="{0}-LicenseServer" ' \
              'Name=instance-state-code,Values=0,16,32,64,80'.format(vpc_name)
    shell_output = _shell_command(command)
    instance_info = _parse_json_shell_output(shell_output, "Reservations")

    # If we have exactly one license server, all is good:
    if len(instance_info) == 1:
        lic_server_id     = instance_info[0]["Instances"][0]["InstanceId"]
        lic_server_status = instance_info[0]["Instances"][0]["State"]["Code"]
        return lic_server_id, lic_server_status
    else:
        if log_error:
            logger = logging.getLogger(__name__)
            logger.error('No license server with the name tag "{0}-LicenseServer" has been configured for this VPC. '
                         'Please use the initialize_virtual_private_cloud() function to initialize one.'.format(name))
        return None, None


def _instance_status_code_to_string(status_code):
    translation_dict = {0: "pending",
                        16: "running",
                        32: "shutting-down",
                        48: "terminated",
                        64: "stopping",
                        80: "stopped"}
    return translation_dict[status_code]


def start_license_server(name):
    """"
    Starts an existing but stopped license server in the named virtual private cloud (VPC).

    :param name: Base name used to generate the VPC and its associated resources.

    :return: True is license server is running now. False if it could not be started.
    """

    # Get the current status of the license server
    lic_server_id, lic_server_status = _find_license_server(name)

    if lic_server_id is not None and lic_server_status == 80: # is stopped
        print('Starting license server. This operation can take several minutes.')
        command='aws ec2 start-instances --instance-ids {0}'.format(lic_server_id)
        _shell_command(command)
        command='aws ec2 wait instance-status-ok --instance-ids {0}'.format(lic_server_id)
        _shell_command(command)
        lic_server_id, lic_server_status = _find_license_server(name) # get new status, should be 16

    elif lic_server_id is not None and lic_server_status == 16:
        logger = logging.getLogger(__name__)
        logger.debug("License server is already running.")
        command='aws ec2 wait instance-status-ok --instance-ids {0}'.format(lic_server_id)
        _shell_command(command)

    # Report current status of the license server
    return _license_server_ready(lic_server_id, lic_server_status)


def _license_server_ready(lic_server_id, lic_server_status):
    """
    Will return True if the License Server is Running, False otherwise.
    Also provides additional debugging information regarding current status.
    """
    logger = logging.getLogger(__name__)

    # Make sure that we have a license manager available at all
    if lic_server_id is None:
        return False

    if lic_server_status in [0,32,64]: # Just in case, we should never see this if Lumerical api is being used
        logger.error("License server can't be started because it's status is '{0}'."
                     " Please wait until status change is complete and try again."
                     .format(_instance_status_code_to_string(lic_server_status)))
        return False
    elif lic_server_status == 16: # Running, we wait just in case the user is starting and stopping from AWS Console
        command='aws ec2 wait instance-status-ok --instance-ids {0}'.format(lic_server_id)
        _shell_command(command)
        return True
    elif lic_server_status == 80: # is stopped
        logger.debug("License server is stopped."
                     " Please use the function start_license_server() to start the server.")
        return False
    elif lic_server_status == 48:
        logger.warning("License server is terminated."
                       " Please use the function start_license_server() to start the server.")
        return False
    else:
        logger.error("License server state is unknown.")
        raise VpcConfigurationError("License server state is unknown.") # There should be no other states


def stop_license_server(name):
    """"
    Stops a running license server in the named virtual private cloud (VPC).

    :param name: Base name used to generate the VPC and its associated resources.

    :return: True is license server is in a stopped state now. False if it could not be stopped or the license server does not exist.
    """
    logger = logging.getLogger(__name__)

    # Get the current status of the license server
    lic_server_id, lic_server_status = _find_license_server(name)
    if lic_server_id is None:
        return False

    elif lic_server_status in [0, 32, 64]:
        logger.error("License server can't be stopped because it's status is '{0}'."
                     " Please wait until status change is complete and try again."
                     .format(_instance_status_code_to_string(lic_server_status)))
        return False
    elif lic_server_status == 80:
        print("License server is already stopped.")
    else:
        print("Stopping license server. This operation can take several minutes.")
        command='aws ec2 stop-instances --instance-ids {0}'.format(lic_server_id)
        _shell_command(command)
        # Wait until instance is stopped.
        command='aws ec2 wait instance-stopped --instance-ids {0}'.format(lic_server_id)
        _shell_command(command)
    return True


def _remote_call_via_ssm(instance_id, remote_command, wait=True):
    """
    Use the Amazon Simple System Manager (SSM) to remotely execute a command on a specific instance

    :param instance_id: List of instance ids on which the command will be executed
    :param remote_command: Command line string to execute
    :param wait: Optional, default is True. When True the function will call the command,
    wait for execution to finish, and return STDOUT
    """
    logger = logging.getLogger(__name__)

    aws_command = 'aws ssm send-command --output json --instance-ids {0} --document-name "AWS-RunShellScript"  ' \
                  '--parameters commands="{1}"'.format(instance_id, remote_command)
    shell_output = _shell_command(aws_command)
    command_id = _parse_json_shell_output(shell_output, "Command")["CommandId"]

    # Get the status and result of the command invocation
    aws_command = 'aws ssm list-command-invocations --output json --command-id "{0}" --details'.format(command_id)

    if not wait:
        logger.debug("Check results of command by running:\n    {}".format(aws_command))
        return ''

    while True:
        sleep(2)
        shell_output = _shell_command(aws_command)
        command_output = _parse_json_shell_output(shell_output, "CommandInvocations")
        status = command_output[0]["Status"]
        if not ((status == "Pending") or (status == "In Progress") or (status == "InProgress")):
            break

    command_stdout = command_output[0]["CommandPlugins"][0]["Output"]
    if command_output[0]["Status"] == "Failed":
        logger.error("SSM command returned:\n{}".format(command_stdout))
    return command_stdout


def license_server_status(name):
    """
    Queries the available CAD and solver licences available from the license server in the named virtual
    private cloud (VPC). It remotely executes the command 'lmstat' on the license server and reports the
    results.

    :param name: Base name used to generate the VPC and its associated resources.
    """
    logger = logging.getLogger(__name__)

    # Get the current status of the license server
    lic_server_id, lic_server_status = _find_license_server(name)

    # Make sure that we have a license server is available at all
    if lic_server_id is None:
        logger.warning('No license server associated with the named VPC could be found.')
        return

    # If the license server not running, just report that
    if lic_server_status != 16:
        logger.warning("License server is currently not running. Server status is '{0}'."
                       " Please use the function start_license_server() to start the server."
                       " Once server is running you can then check the status again for additional information."
                       .format(_instance_status_code_to_string(lic_server_status)))
        return

    # Also make sure that we wait for the server to be initialized
    print('Waiting for license server response...')
    command='aws ec2 wait instance-status-ok --instance-ids {0}'.format(lic_server_id)
    _shell_command(command)

    # Make two calls to lumstat. First with -i to see all the available licenses
    remote_command='/opt/lumerical/lumerical-flexlm/lmutil lmstat -i'
    logger.debug('Using SSM to issue command "{0}" on instance with ID {1}'.format(remote_command,lic_server_id))
    command_output = _remote_call_via_ssm(lic_server_id, remote_command)
    print(command_output)

    # Second call to -a to see what is actually available (and what is currently used)
    remote_command='/opt/lumerical/lumerical-flexlm/lmutil lmstat -a'
    logger.debug('Using SSM to issue command "{0}" on instance with ID {1}'.format(remote_command,lic_server_id))
    command_output = _remote_call_via_ssm(lic_server_id, remote_command)
    print(command_output)

    return


def update_license_file(name, license_file_local_path):
    logger = logging.getLogger(__name__)

    license_file_local_path = expanduser(license_file_local_path) # allow ~/
    if not isfile(license_file_local_path):
        logger.error('A license file with the name "{0}" could not be found.'.format(license_file_local_path))
        return
    with open(license_file_local_path, 'rb') as lic_file:
        license_file_content = lic_file.read().decode('ascii')

    lic_server_id, _ = _find_license_server(name)
    if lic_server_id is None:
        return

    f_path = "/opt/lumerical/lumerical-flexlm/licenses/LUMERICL/AWSLicenseFile.lic"
    _ssm_transfer_files(lic_server_id, [(f_path, license_file_content.encode()), ])


def configure_compute_instance(name, instance_type=None, max_storage_in_gb=60):
    """
    Creates an elastic cloud computing (EC2) launch configuration for the compute nodes in the named virtual
    private cloud (VPC). Refer to the Amazon web services (AWS) site for machine instance types and their
    minimum allowed storage sizes.

    :param name:              Base name used to create the VPC and its associated resources.
    :param instance_type:     Amazon EC2 instance type (e.g. t2.medium)
    :param max_storage_in_gb: Maximum disk storage in gigabytes.
    """
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)
    vpc_name = _get_vpc_name(name)

    command = 'aws ec2 describe-vpcs --filter "Name=tag-value,Values={0},{1}" --query Vpcs[0]'.format(vpc_name,stack_name)
    shell_output = _shell_command(command)
    if vpc_name not in shell_output and stack_name not in shell_output:
        logger.error('A VPC with the name "{0}" has not yet been created.'.format(vpc_name))
        return

    lic_server_id, lic_server_status = _find_license_server(name, log_error=False)
    if lic_server_id is None or license_server_status in [None, 48]:
        return

    params = _construct_parameters(setup_compute_nodes='true', num_compute_nodes=0, compute_node_instance_type=instance_type, compute_node_max_storage=max_storage_in_gb)
    _update_stack(name, params)
    compute_instance_status(name)

    return


def _wait_for_running_instances(name, num_instances, workgroup_id):
    """
    Waits for all the compute node instances in the given workgroup ID to be lauched and running (status O.K.)
    There is a brief period of time when a call to `wait stack-update-complete` returns but the instances are
    not visible via `describe-instances` for a few seconds, so an additional wait is necessary.

    :param name:          Base name used to generate the VPC and its associated resources.
    :param num_instances: Number of compute node instances that was launched.
    :param workgroup_id:  Workgroup ID generated when starting the compute nodes.
    """
    logger = logging.getLogger(__name__)

    poll_time = 10  # seconds
    max_num_polls = 8
    print("Waiting for new instances. This operation can take several minutes.")
    for i in range(int(max_num_polls)):
        instances = _get_starting_instances(name, workgroup_id)
        if len(instances) == num_instances:
            print("{} instances found with WorkgroupId {}."
                  " Waiting for instances to complete initialization.".format(num_instances, workgroup_id))
            instance_ids = _get_instances_ids(instances)
            command = 'aws ec2 wait instance-status-ok --instance-ids {0}'.format(' '.join(instance_ids))
            _shell_command(command)
            return
        else:
            sleep(poll_time)

    logger.error("Failed to find {} running compute instances".format(num_instances))
    return


def _find_autoscaling_group(name):
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)

    # Figure out what the current desired capacity is
    command = '''aws autoscaling describe-auto-scaling-groups --output json --query "AutoScalingGroups[?contains(AutoScalingGroupName,'{0}')]"'''.format(stack_name+'-ComputeNodeGroup')
    shell_output = _shell_command(command)
    autoscaling_groups = _parse_json_shell_output(shell_output)

    if len(autoscaling_groups) == 1:
        cur_capacity = autoscaling_groups[0]["DesiredCapacity"]
        group_name = autoscaling_groups[0]["AutoScalingGroupName"]
    else:
        logger.error("No unique autoscaling group found."
                     " Please run configure_compute_instance() or initialize_virtual_private_cloud()"
                     " to re-initialize.")
        return None, None

    return group_name, cur_capacity


def start_compute_instances(name, num_instances, vnc_password=None):
    """
    Launches the specified number of compute node instances and tags them with a unique workgroup ID. If no VNC
    password for the compute nodes is specified, a random password is generated and reported. The VNC password
    is not saved and it is up to the caller to collect it for future reference.


    :param name:          Base name used to generate the virtual private cloud (VPC) and its associated resources.
    :param num_instances: Number of compute node instances to be launched.
    :param vnc_password:  Password used for VNC access to all the compute node instances.

    :returns: Workgroup ID string.
    """
    logger = logging.getLogger(__name__)

    if num_instances < 1:
        error_msg = "Number of instances must be at least 1."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    stack_name = _get_stack_name(name)
    command = '''aws autoscaling describe-launch-configurations --output json --query "LaunchConfigurations[?contains(LaunchConfigurationName,'{0}')]"'''.format(stack_name)
    shell_output = _shell_command(command)
    launch_configs = _parse_json_shell_output(shell_output)
    if not launch_configs or len(launch_configs) == 0:
        error_msg = "No launch configuration for the compute nodes has been defined. " \
                    "Please use the configure_compute_instance() function to create one."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    timestamp = datetime.datetime.fromtimestamp(time.time())

    _workgroup_id = name+'-'+timestamp.strftime('%Y%m%d-%H%M%S')

    # We should have exactly one matching auto-scaling group.
    group_name, cur_capacity = _find_autoscaling_group(name)
    if group_name is None:
        error_msg = "No auto auto-scaling group could be found."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    num_total_instances = cur_capacity + num_instances

    # Manually set the new workgroup ID and the new desired capacity.
    command = "aws autoscaling create-or-update-tags --tags ResourceId={0},ResourceType=auto-scaling-group,Key=lum:workgroup-id,Value={1},PropagateAtLaunch=true".format(group_name, _workgroup_id)
    _shell_command(command)
    command = "aws autoscaling set-desired-capacity --auto-scaling-group-name {0} --desired-capacity {1}".format(group_name, num_total_instances)
    _shell_command(command)

    if not vnc_password:
        num_chars = 12
        vnc_password = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(num_chars))

    _wait_for_running_instances(name, num_instances, _workgroup_id)
    _compute_instances_configure_post_launch(name, _workgroup_id, vnc_password)

    # Reporting
    print('Compute instances started.')
    print('  Time                       : {0}'.format(timestamp.strftime('%Y-%m-%d %H:%M:%S')))
    print('  Workgroup ID               : {0}'.format(_workgroup_id))
    print('  Number of instances started: {0}'.format(num_instances))
    print('  Initial VNC password is    : {0}'.format(vnc_password))
    _list_all_instances(name)
    return _workgroup_id


def terminate_all_instances(name):
    """
    Terminates all compute instances that belong to the named virtual private cloud (VPC). Workgroup ID is ignored.

    :param name: Base name used to create the VPC and its associated resources.
    """
    logger = logging.getLogger(__name__)

    group_name, cur_capacity = _find_autoscaling_group(name)
    if group_name is None:
        error_msg = "No auto auto-scaling group could be found."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Terminating happens automatically if we just set the desired capacity to zero
    if cur_capacity > 0:
        print("Please wait while we terminate all instances. This operation can take several minutes.")

        command = "aws autoscaling set-desired-capacity --auto-scaling-group-name {0} --desired-capacity 0".format(group_name)
        _shell_command(command)

    else:
        logger.warning("There are no running instances. Nothing to do.")
    return


def compute_instance_status(name):
    """
    Prints the status of the compute node configuration and lists all instances with that configuration.

    :param name: Base name used to generate the virtual private cloud (VPC) and its associated resources.
    """
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)
    command = '''aws autoscaling describe-launch-configurations --output json --query "LaunchConfigurations[?contains(LaunchConfigurationName,'{0}')]"'''.format(stack_name)
    shell_output = _shell_command(command)
    launch_configs = _parse_json_shell_output(shell_output)
    if not launch_configs or len(launch_configs) == 0:
        logger.error("No launch configuration for the compute instances has been defined. "
                     "Please use configure_compute_instance() to configure instances.")
        return

    for launch_config in launch_configs:
        instance_name = launch_config["LaunchConfigurationName"]
        instance_type = launch_config["InstanceType"]
        disk_space = launch_config["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"]
        print('Launch configuration {0}'.format(instance_name))
        print('  Instance Type       : {0}'.format(instance_type))
        print('  Max. Storage        : {0}GB'.format(disk_space))

    #  And now we also list all running instances (if any)
    _list_all_instances(name)
    return


def _get_all_instances(name):
    vpc_name = _get_vpc_name(name)
    command = 'aws ec2 describe-instances --output json --filters Name=tag-value,Values="{0}-Node"'.format(vpc_name)
    shell_output = _shell_command(command)
    instance_info = _parse_json_shell_output(shell_output, "Reservations")

    instances=[]
    for reservation in instance_info:
        for instance in reservation["Instances"]:
            # This will print will output the value of the Dictionary key 'InstanceId'
            instance_id = instance["InstanceId"]
            instance_state = instance["State"]["Name"]
            instance_workgroup_id = [tag['Value'] for tag in instance["Tags"] if tag['Key'] == 'lum:workgroup-id']
            if 'PrivateIpAddress' in instance:
                instance_ip = instance['PrivateIpAddress']
            else:
                instance_ip = None
            if 'PublicIpAddress' in instance:
                instance_public_ip = instance['PublicIpAddress']
            else:
                instance_public_ip = None

            # Remove the brackets if exactly one workgroup_id (as should be the case)
            if len(instance_workgroup_id) == 1:
                instance_workgroup_id = instance_workgroup_id[0]

            instances.append({"Id": instance_id, "State": instance_state, "WorkgroupId": instance_workgroup_id,
                              "IP": instance_ip, "PublicIP": instance_public_ip})
    return instances


def _list_all_instances(name):
    """
    Lists all the instance that belong to the VPC/stack

    :param name: Base name used when creating the stack. Used to generate names for the vpc etc.
    """
    logger = logging.getLogger(__name__)

    instances = _get_all_instances(name)

    if len(instances) == 0:
        logger.warning('No compute instances have been started in a VPC with the name "{0}".'.format(name))
        return

    print('Instances:')
    for instance in instances:
        print('  Instance {0:20} | Status: {1:15} | WorkgroupId: {2:30} | IP: {3:15} | Public IP: {4:15}'
              .format(instance["Id"], instance["State"], instance["WorkgroupId"],
                      str(instance['IP']), str(instance['PublicIP'])))


def _s3_isfile(s3_uri):
    """
    Checks that a project file exists in the provided S3 URI.
    Note that, when calling s3 ls with a file path only the file is listed.
    If the file does not exist at that location, nothing is listed.

    :param s3_uri: File path in an S3 bucket
    :returns: True if the file exists in the bucket.
    """
    logger = logging.getLogger(__name__)
    command = "aws s3 ls {}".format(s3_uri)
    try:
        ret = _shell_command(command, raise_exception=True)
    except CalledProcessError as exc:
        if "NoSuchBucket" in exc.output:
            logger.error(exc.output)
            return False
        else:
            raise exc
    ret = ret.splitlines()
    if len(ret) == 0:
        logger.error('File: "{}" does not exits'.format(s3_uri))
        return False
    elif len(ret) > 1:
        logger.error('S3 URI: "{}" appears to be a folder. A file was expected.'.format(s3_uri))
        return False
    return True


def run_parameter_sweep(name, workgroup_id, s3_uri, sweep_name=None, terminate_instances=True, wait=False):
    """
    Runs a parameter sweep or optimization. The project file must be placed in a simple storage service (S3) bucket ahead
    of time and referenced here using a unique resource identifier (URI). The specified parameter sweep or optimization
    will be run on all the compute node instances associated with the provided workgroup ID. If a sweep or optimization
    object name is not passed to this function, all the sweeps and optiizations defined in the project file are run.

    :param name:         Base name used to generate the virtual private cloud (VPC) and its associated resources.
    :param workgroup_id: Workgroup ID generated when starting the compute nodes.
    :param s3_uri:       S3 URI with the desired project file name.
    :param sweep_name:   Name of the parameter sweep or optimization inside the project file to be run.
    :param terminate_instances: Default is True. When True all instances with the WorkGroup ID will be terminated.
    :param wait:         True to wait for parameter sweep or optimization to complete before exiting.
    """
    logger = logging.getLogger(__name__)

    start_license_server(name)

    lic_server_id, lic_server_status = _find_license_server(name)
    if not _license_server_ready(lic_server_id, lic_server_status):
        logger.error("Cannot find running license server.")
        return

    running_instances = _get_ready_compute_nodes(name, workgroup_id)
    if len(running_instances) == 0:
        logger.error('No compute nodes with the workgroup ID "{1}" are running in a VPC with the name "{0}".'.format(name, workgroup_id))
        return

    s3_bucket, file_name = split(s3_uri)
    _, ext = splitext(file_name)
    ext = ext.lower()

    if not _s3_isfile(s3_uri):
        return

    if ext == '.fsp':
        cad_path = FDTD_CAD
    elif ext == '.ldev':
        cad_path = DAVICE_CAD
    elif ext == '.lms':
        cad_path = MODE_CAD
    else:
        logger.error("File type not supported.")
        raise VpcConfigurationError("File type not supported.")

    working_dir = "/home/ec2-user/{workgroup_id}".format(workgroup_id=workgroup_id)
    lum_script_path = "{working_dir}/aws_run_sweep.lsf".format(working_dir=working_dir)
    bash_script_path = "{working_dir}/aws_run_sweep.sh".format(working_dir=working_dir)
    log_file_path = "{working_dir}/{workgroup_id}.log".format(workgroup_id=workgroup_id, working_dir=working_dir)
    log_file_path_s3 = "{s3_bucket}/{workgroup_id}.log".format(workgroup_id=workgroup_id, s3_bucket=s3_bucket)

    # file copied from s3 to local
    lum_script = "load('{s3_uri}');\n".format(s3_uri=s3_uri)
    lum_script += "runsweep;\n" if sweep_name is None else "runsweep('{sweep_name}');\n".format(sweep_name=sweep_name)
    lum_script += "save;\n"

    # This extra shell script is necessary to redirect stderr and segfaults to stdout
    bash_script = ("{cad_path} -logfile {log_file_path} -run {lum_script_path} -exit &>> {log_file_path};"
                  ).format(cad_path=cad_path, lum_script_path=lum_script_path, workgroup_id=workgroup_id, log_file_path=log_file_path)
    bash_script = "{ " + bash_script + " }" + " &>> {log_file_path}; exit 0".format(log_file_path=log_file_path)

    # since ssm commands are run as root we need to run fdtd-solutions as ec2-user else the windows renders all white
    run_command = ("mkdir -p {working_dir};"
                   "echo $(date)    Starting simulations. &>> {log_file_path};"
                   "chown -R ec2-user {working_dir};"
                   "chmod +x {bash_script_path} &>> {log_file_path};"
                   "export DISPLAY=:0 &>> {log_file_path};"
                   "su -c {bash_script_path} ec2-user &>> {log_file_path};"
                   "aws s3 cp {log_file_path} {log_file_path_s3};"
                   "echo DONE;"
                   ).format(bash_script_path=bash_script_path, log_file_path=log_file_path, log_file_path_s3=log_file_path_s3, working_dir=working_dir)
    if terminate_instances:
        run_command += "python /opt/lumerical/helper_scripts/terminateJobInstances.py;"

    instances = _get_ready_compute_nodes(name, workgroup_id)
    instance_ids = _get_instances_ids(instances)
    if len(instance_ids) == 0:
        logger.error("No running compute node instances with workgroup ID {} could be found.".format(workgroup_id))
        raise RuntimeError("No running compute node instances with workgroup ID {} could be found.".format(workgroup_id))

    root_node = instances[0]
    _ssm_transfer_files(root_node['Id'], [(lum_script_path, lum_script.encode()),
                                          (bash_script_path, bash_script.encode())])

    print("Parameter sweep has been launched. You can monitor its progress via a VNC connection to: {}".format(root_node['PublicIP']))
    _remote_call_via_ssm(root_node['Id'], run_command, wait=wait)

    return


def kill_job(name, workgroup_id):
    """
    Terminates all the compute node instances tagged with the specified workgroup ID.

    :param name:         Base name used to generate the virtual private cloud (VPC) and its associated resources.
    :param workgroup_id: Unique identifier returned when the compute node instances were launched.
    """
    logger = logging.getLogger(__name__)

    instances = _get_all_instances(name)
    if len(instances) == 0:
        logger.warning('No compute instances have been started in a VPC with the name "{0}".'.format(name))
        return

    # Find the running compute node instance ids with the matching workgroup_id.
    filtered_instance_ids = [x['Id'] for x in instances if x['WorkgroupId'] == workgroup_id and x['State'] == 'running']

    if len(filtered_instance_ids) == 0:
        logger.warning('No running instances associated with the workgroup ID {0} were found.'.format(workgroup_id))
        return

    # Iterate over the compute node instances and terminate each one.
    for instance_id in filtered_instance_ids:
        print('Terminating instance {0}'.format(instance_id))
        command = 'aws autoscaling terminate-instance-in-auto-scaling-group --instance-id {0} --should-decrement-desired-capacity'.format(instance_id)
        _shell_command(command)

    command = 'aws ec2 wait instance-terminated --instance-ids {0}'.format(' '.join(filtered_instance_ids))
    _shell_command(command)

    return


def stop_job(name, workgroup_id):
    """
    Termintes all the compute node instances tagged with the specified workgroup ID. Prior to termination, a SIGTERM signal
    is issued to all running simulations in all compute node instances. All running simulations are given a time window of
    20 seconds to quit and save.

    :param name:         Base name used to generate the virtual private cloud (VPC) and its associated resources.
    :param workgroup_id: Unique identifier returned when the compute node instances were launched.
    """
    logger = logging.getLogger(__name__)

    instances = _get_all_instances(name)
    if len(instances) == 0:
        logger.warning('No compute instances have been started in a VPC with the name "{0}".'.format(name))
        return

    # Find the running instance ids with the matching workgroup_id.
    filtered_instance_ids = [x['Id'] for x in instances if x['WorkgroupId'] == workgroup_id and x['State'] == 'running']

    if len(filtered_instance_ids) == 0:
        logger.warning('No running instances with workgroup ID {0} were found'.format(workgroup_id))
        return

    # Send the SIG_TERM signal to all products using SSM
    kill_commands = 'pkill -f /opt/lumerical/'
    _remote_call_via_ssm(' '.join(filtered_instance_ids), kill_commands)

    # Iterate over all the running instances and terminate each one.
    for instance_id in filtered_instance_ids:
        print('Terminating instance {0} ...'.format(instance_id))
        command = 'aws autoscaling terminate-instance-in-auto-scaling-group --instance-id {0} --should-decrement-desired-capacity'.format(instance_id)
        _shell_command(command)

    command = 'aws ec2 wait instance-terminated --instance-ids {0}'.format(' '.join(filtered_instance_ids))
    _shell_command(command)

    return


def _get_starting_instances(name, workgroup_id):
    instances = _get_all_instances(name)
    running_instances = list(filter(lambda x: (x['State'] in ['running', 'pending', 'rebooting']) and
                                              (x['WorkgroupId'] == workgroup_id), instances))
    return running_instances


def _get_ready_compute_nodes(name, workgroup_id):
    instances = _get_all_instances(name)
    instances = filter(lambda x: (x['State'] == 'running') and (x['WorkgroupId'] == workgroup_id), instances)
    return list(instances)


def _get_instances_ids(instances):
    return list(map(lambda x: x['Id'], instances))


def _get_instances_ips(instances):
    return list(map(lambda x: x['IP'], instances))


def _compute_instances_configure_post_launch(name, workgroup_id, vnc_password=None):
    logger = logging.getLogger(__name__)
    logger.debug("Applying settings to running compute nodes ...")

    # check license server exists
    lic_server_id, _ = _find_license_server(name)
    if lic_server_id is None:
        return
    # get license server ip
    aws_command = 'aws ec2 describe-instances --output json' \
                  ' --filters Name=instance-state-code,Values=0,16,32,64,80 Name=tag-value,Values={0}-LicenseServer' \
                  ' --query "Reservations[0].Instances[*].PrivateIpAddress"' \
                  .format(_get_vpc_name(name))
    shell_output = _shell_command(aws_command)
    license_server_ip = json.loads(shell_output)
    assert len(license_server_ip) == 1
    license_server_ip = license_server_ip[0]

    # get all running compute nodes with workgroup_id
    running_instances = _get_ready_compute_nodes(name, workgroup_id)
    if len(running_instances) == 0:
        logger.error("Called configure compute nodes post launch with no running instances.")
        return

    instance_ids = _get_instances_ids(running_instances)
    instance_id_str = ' '.join(instance_ids)
    instance_ips = _get_instances_ips(running_instances)

    if vnc_password:
        _compute_instances_set_vnc_password(instance_id_str, vnc_password)

    command = _get_command_no_password_compute_nodes()
    ret = _remote_call_via_ssm(instance_id_str, command)
    print(ret)

    # Separate call for each product since otherwise the command line quickly becomes too long
    _ssm_transfer_files(instance_id_str, [(FDTD_INI,
                                           _get_fdtd_config(license_server_ip, instance_ips))])
    _ssm_transfer_files(instance_id_str, [(MODE_INI,
                                           _get_mode_config(license_server_ip, instance_ips))])
    _ssm_transfer_files(instance_id_str, [(DEVICE_INI,
                                           _get_device_config(license_server_ip, instance_ips))])


def _compute_instances_set_vnc_password(instance_ids, vnc_password, username='ec2-user'):
    vnc_password = vnc_password.replace('"', '\\"')
    command = (
        "echo {vnc_password} | vncpasswd -f > /home/{username}/.vnc/passwd;"
        "chown -R {username}:{username} /home/{username}/.vnc;"
        "chmod 600 /home/{username}/.vnc/passwd;"

        "systemctl enable vncserver@:0.service 2>&1;"
        "systemctl start vncserver@:0.service 2>&1;"
        ).format(vnc_password=vnc_password, username=username)
    ret = _remote_call_via_ssm(instance_ids, command)
    logger = logging.getLogger(__name__)
    logger.debug("Command Returned: {}".format(ret))


def _get_command_no_password_compute_nodes(user = 'ec2-user'):
    pri_key, pub_key = _generate_rsa_key()

    ssh_dir = "/home/{user}/.ssh".format(user=user)
    command = "base64 --decode <<< {private_key} > {ssh_dir}/id_rsa;" \
              "base64 --decode <<< {public_key} >> {ssh_dir}/authorized_keys;"

    command += "echo StrictHostKeyChecking no >> {ssh_dir}/config;" \
               "chmod 400 {ssh_dir}/config;" \
               "rm -f {ssh_dir}/known_hosts;" \
               "chown {user} {ssh_dir}/config;"

    command = command.format(ssh_dir=ssh_dir, user=user,
                             private_key=b64encode(pri_key.encode()).decode('ascii'),
                             public_key=b64encode(('\n'+pub_key+' compute-nodes\n').encode()).decode('ascii'))
    return command


def _generate_rsa_key():
    # generate private/public key pair
    key = rsa.generate_private_key(backend=default_backend(), public_exponent=65537, key_size=2048)

    # get public key in OpenSSH format
    public_key = key.public_key().public_bytes(serialization.Encoding.OpenSSH,
                                               serialization.PublicFormat.OpenSSH)

    # get private key in PEM container format
    pem = key.private_bytes(encoding=serialization.Encoding.PEM,
                            format=serialization.PrivateFormat.TraditionalOpenSSL,
                            encryption_algorithm=serialization.NoEncryption())

    # decode to printable strings
    private_key_str = pem.decode('utf-8')
    public_key_str = public_key.decode('utf-8')

    return private_key_str, public_key_str


def _gzip_data(file_data):
    # file_data MUST be encoded as a byte-object prior to passing (use .encode())
    # In python 3 it is very easy:
    if sys.version_info > (3, 0):
        return gzip.compress(file_data, compresslevel=9)
    else:
        fgz = StringIO.StringIO()
        with gzip.GzipFile(mode='wb', compresslevel=9, fileobj=fgz) as gzip_obj:
            gzip_obj.write(file_data)
        ret = str(fgz.getvalue())
        fgz.close()

        return ret


def _ssm_transfer_files(instance_ids, file_path_content_pair_list, use_gzip=True):
    """
    Transfers all files in a given list to all instances in a given list over SSM

    :param instance_ids: list of Amazon ec2 instances (separated by spaces) that the files will be transfered to
    :param file_path_data_pair_list: a list of file and file-content pairs
    """
    command = str()
    for file_path, file_data in file_path_content_pair_list:
        dst_dir, _ = split(file_path)

        # We can either use gzip (to reduce the size of the parameter) or not
        if use_gzip:
            file_data_base64 = b64encode(_gzip_data(file_data)).decode('ascii')
            command += "mkdir -p {output_dir};" \
                       "base64 --decode <<< {file_data_base64} | gunzip > {file_path};".format(file_data_base64=file_data_base64, file_path=file_path, output_dir=dst_dir)

        else:
            file_data_base64 = b64encode(file_data).decode('ascii')

            command += "mkdir -p {output_dir};" \
                       "base64 --decode <<< {file_data_base64} > {file_path};" \
                       .format(file_data_base64=file_data_base64, file_path=file_path, output_dir=dst_dir)

    ret = _remote_call_via_ssm(instance_ids, command)
    print(ret)


def _get_common_config(license_server_hostname, license_server_port=27000):
    # General configuration
    general_config = '\n[General]\nshowWelcome=false\n'
    # License configuration
    license_config = "\n[license]\ntype=flex\nflexserver\host={0}@{1}\n".format(license_server_port, license_server_hostname)

    return general_config + license_config


def _generate_engines_line(engine_name, solver_name, exe_name, exe_path, mpi_path, compute_node_hostname_array, n_processes, n_threads):
    engine_list = '{engine_name}=<engines>'.format(engine_name=engine_name)
    engine_sub_config = "<engine><name>{compute_node_hostname}</name><host>{compute_node_hostname}</host><nProc>{n_processes}</nProc><nThread>{n_threads}</nThread><id>1</id><active>true</active><solverName>{solver_name}</solverName><exeName>{exe_name}</exeName><multiHosts>true</multiHosts><multiProcs>true</multiProcs><advanced><runType>Custom</runType><mpiPath>{mpi_path}</mpiPath><exePath>{exe_path}</exePath><useBinding>false</useBinding><logAll>true</logAll><extraMpiOptions></extraMpiOptions><suppressDefaultMPI>false</suppressDefaultMPI><suppressDefaultEngine>false</suppressDefaultEngine><extraOptions></extraOptions><bypassLocalMpi>false</bypassLocalMpi></advanced></engine>"

    for compute_node_hostname in compute_node_hostname_array:
        engine_list += engine_sub_config.format(compute_node_hostname=compute_node_hostname, n_processes=n_processes, n_threads=n_threads, engine_name=engine_name, solver_name=solver_name, exe_name=exe_name, exe_path=exe_path, mpi_path=mpi_path)
    engine_list += '</engines>\n'
    return engine_list


def _get_fdtd_config(license_server_hostname, compute_node_hostname_array):
    # common
    configuration_ini = _get_common_config(license_server_hostname)
    # Start engine configurations
    configuration_ini += "\n[jobmanager]\n"
    # FDTD Engine configuration
    fdtd_mpiexec = "mpiexec" # Assume user has Open MPI installed
    configuration_ini += _generate_engines_line("FDTD", "FDTD", "fdtd-engine-ompi-lcl", FDTD_BIN + "/fdtd-engine-ompi-lcl", fdtd_mpiexec, compute_node_hostname_array, n_processes=1, n_threads=0)

    return configuration_ini.encode()


def _get_mode_config(license_server_hostname, compute_node_hostname_array):
    # common
    configuration_ini = _get_common_config(license_server_hostname)
    # Start engine configurations
    configuration_ini += "\n[jobmanager]\n"
    # MODE Engines configuration
    mode_mpiexec = "mpiexec" # Assume user has Open MPI installed
    configuration_ini += _generate_engines_line("Eigensolver", "Eigensolver", "fd-engine",                MODE_BIN + "/fd-engine",                mode_mpiexec, compute_node_hostname_array, n_processes=1, n_threads=0)
    configuration_ini += _generate_engines_line("varFDTD",     "varFDTD",     "varfdtd-engine-ompi-lcl",  MODE_BIN + "/varfdtd-engine-ompi-lcl",  mode_mpiexec, compute_node_hostname_array, n_processes=1, n_threads=0)
    configuration_ini += _generate_engines_line("EME",         "EME",         "eme-engine-ompi-lcl",      MODE_BIN + "/eme-engine-ompi-lcl",      mode_mpiexec, compute_node_hostname_array, n_processes=1, n_threads=0)

    return configuration_ini.encode()


def _get_device_config(license_server_hostname, compute_node_hostname_array):
    # common
    configuration_ini = _get_common_config(license_server_hostname)
    # Start engine configurations
    configuration_ini += "\n[jobmanager]\n"
    # DEVICE Engines configurations
    device_mpiexec = "mpiexec" # Assume user has Open MPI installed
    configuration_ini += _generate_engines_line("Charge%20Transport%20Solver",            "Charge Transport Solver",            "device-engine-ompi-lcl",  DEVICE_BIN + "/device-engine-ompi-lcl", device_mpiexec, compute_node_hostname_array, n_processes=1, n_threads=0)
    configuration_ini += _generate_engines_line("Heat%20Transport%20Solver",              "Heat Transport Solver",              "thermal-engine-ompi-lcl", DEVICE_BIN + "/thermal-engine-ompi-lcl", device_mpiexec, compute_node_hostname_array, n_processes=1, n_threads=0)
    configuration_ini += _generate_engines_line("Electromagnetic%20Propagation%20Solver", "Electromagnetic Propagation Solver", "dgtd-engine-ompi-lcl",    DEVICE_BIN + "/dgtd-engine-ompi-lcl", device_mpiexec, compute_node_hostname_array, n_processes=1, n_threads=0)

    return configuration_ini.encode()


def remove_virtual_private_cloud(name, force=False):
    """
    Deletes the named virtual private cloud (VPC) and all of its associated resources including the network interface.
    This will invalidate any node locked licenses associated with the MAC address of the network interface.

    :param name: Base name used to generate the virtual private cloud (VPC) and its associated resources.
    :param force: True to bypass prompt requests and force the removal of the VPC.
    """
    logger = logging.getLogger(__name__)

    stack_name = _get_stack_name(name)
    vpc_name = _get_vpc_name(name)
    command = 'aws ec2 describe-vpcs --filter "Name=tag-value,Values={0},{1}" --query Vpcs[0]'.format(vpc_name, stack_name)
    shell_output = _shell_command(command)
    if vpc_name not in shell_output and stack_name not in shell_output:
        logger.warning("A VPC with the name '{0}' could not be found.".format(name))
        return

    if not force:
        print(("WARNING: This operation will erase the virtual private cloud '{0}' and invalidate any "
              "license associated with its network interface. The operation is final and cannot be undone.\n"
              "Do you wish to proceed (yes|no)?").format(name))

        # To support both python 2.X and 3.X
        get_input = input
        if sys.version_info[:2] <= (2, 7):
            get_input = raw_input

        choice = get_input().lower()
        if choice != "yes":
            print("Command cancelled.")
            return

    description, stack_id, vpc_id, subnet_cidr, lic_ami_id, ami_id, account_no, region, subnet_id, security_grp_id, parameters=_get_stack_info(stack_name)
    _delete_endpoints(vpc_id, region, "ssm", subnet_id, security_grp_id)
    _delete_endpoints(vpc_id, region, "ec2messages", subnet_id, security_grp_id)

    command = 'aws cloudformation delete-stack --stack-name {0}'.format(stack_name)
    _shell_command(command)
    print("Removing virtual private cloud with the name '{0}'. This operation can take several minutes and will be processed in the background.".format(name))

    # Wait for confirmation that VPC has been deleted.
    command = 'aws cloudformation wait stack-delete-complete --stack-name {0}'.format(stack_name)
    error_msg = 'The requested VPC could not be successfully deleted.'
    _shell_command(command, error_msg)

    return

configure_logging()
