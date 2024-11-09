from web3 import Web3

# Replace 'YOUR_PROVIDER_URL' with your actual Ethereum provider URL
w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/18bfb424153047309e3c6d33ddaff587'))

if w3.is_connected():  # Use is_connected instead of isConnected
    print("Connected to Ethereum blockchain")
else:
    print("Failed to connect to Ethereum blockchain")
