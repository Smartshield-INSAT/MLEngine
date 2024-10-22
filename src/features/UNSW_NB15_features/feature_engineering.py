import pandas as pd

top_prop_categories = ['tcp', 'udp', 'unas', 'arp', 'ospf', 'sctp']
top_service_categories = ['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']
top_state_categories = ['INT', 'FIN', 'CON', 'REQ', 'RST']
log_features = ['smean', 'spkts', 'dpkts', 'sloss', 'dloss', 'response_body_len', 'sinpkt', 'dinpkt', 'sload', 'dload']