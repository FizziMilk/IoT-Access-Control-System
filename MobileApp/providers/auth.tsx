//auth.tsx
import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import * as SecureStore from 'expo-secure-store';


type AuthContextType = {
    token: string | null;
    signIn: (token: string) => Promise<void>;
    signOut: () => Promise<void>;
    isLoading: boolean;
};

const AuthContext = createContext<AuthContextType>({
    token: null,
    signIn: async () => {},
    signOut: async() => {},
    isLoading: true,
});

//so useContext and AuthContext don't need to be imported each time
export const useAuth = (): AuthContextType => useContext(AuthContext);

//Define the type of props the AuthProvider will receive (any react node child)
type AuthProviderProps = {
    children: ReactNode;
};
//Main function to provide authorization to any component that needs it
export const AuthProvider: React.FC<AuthProviderProps> = ({ children}) => {
    const [token, setToken] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);

    // Loads a stored token the first time AuthProvider is accessed (if exists)
    useEffect(() => {
        (async () => {
            const storedToken = await SecureStore.getItemAsync('accessToken');
            setToken(storedToken);
            setIsLoading(false);
        })();
    }, []);
    // Takes a token as argument and saves it
    const signIn = async (newToken: string) => {
        await SecureStore.setItemAsync('accessToken', newToken);
        setToken(newToken);
    }
    // Deletes existing token
    const signOut = async () => {
        await SecureStore.deleteItemAsync('accessToken');
        setToken(null);
    };
    // 
    return (
        <AuthContext.Provider value ={{ token, signIn, signOut, isLoading}}>
            {children}
        </AuthContext.Provider>
    );
};
