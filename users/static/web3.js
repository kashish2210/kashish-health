// Check if MetaMask is installed
window.addEventListener('load', async () => {
    const walletAddressElement = document.getElementById('wallet-address');
    if (typeof window.ethereum !== 'undefined') {
        console.log('MetaMask is installed!');
        const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
        if (accounts.length > 0) {
            const walletAddress = accounts[0];
            walletAddressElement.textContent = `Wallet: Connected (${walletAddress})`;
        } else {
            walletAddressElement.textContent = 'Wallet: Not connected';
        }
    } else {
        walletAddressElement.textContent = 'Wallet: Not connected (MetaMask not installed)';
    }
});
