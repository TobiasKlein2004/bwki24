import { StyleSheet, Text, View } from 'react-native';

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
    },
    camera: {
        flex: 1,
    },
    blackOverlayBottom: {
        position: 'absolute',
        bottom: 0,
        width: '100%',
        height: '30%',
        backgroundColor: 'rgba(0,0,0,0.8)'
    },
    blackOverlayTop: {
        position: 'absolute',
        top: 0,
        width: '100%',
        height: '20%',
        backgroundColor: 'rgba(0,0,0,0.8)'
    },
    button: {
        position: 'absolute',
        bottom: '10%',
        width: '100%',
        display: 'flex',
        alignItems: 'center'
    },
    cameraImage: {
        width: 100,
        height: 100,
        resizeMode: 'stretch'
    },
    text: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
    },
    preview: {
        background: 'transparent',
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
    },
    previewImage: {
        flex: 1
    }
});

module.exports = {styles};