import os
import pickle
import re

import numpy as np
import pandas as pd


class Precificacao():
    
    def __init__( self ):
        
        self.quarto_area_construida_scaler = pickle.load(open("./parametros/parametrosquartos_por_areaconstruida.pkl",'rb')) 
        self.preco_por_metro_scaler = pickle.load(open('./parametros/parametrospreco_por_metro.pkl','rb'))
        self.bairro_por_area_scaler = pickle.load(open('./parametros/parametrosbairro_por_area.pkl','rb')) 
        self.bairro_scaler = pickle.load(open('./parametros/parametrosbairro_scaler.pkl','rb')) 
        self.banheiros_scaler = pickle.load(open('./parametros/parametrosbanheiros_scaler.pkl','rb')) 
        self.vagas_scaler = pickle.load(open('./parametros/parametrosvagas_scaler.pkl','rb')) 
        self.model = pickle.load(open('./model/model_price.pkl','rb'))  
        
        
        
    
    #load
    def ler_csv(self, csv):
        df_raw = os.path.join(self.INPUT_DIR, csv)  
        df = pd.read_csv(df_raw, sep=",")
        
        return df
    
    def data_rearrangement( self, df):
    

        # Rearrumando o Bairro que estava em coluna errada, replace Vagas pela mais comum
        df.loc[df.Vagas == 'Morros', 'Bairro' ] = 'Morros'


        df.loc[df.Vagas == 'Ininga', 'Bairro' ] = 'Ininga'
        df.loc[df.Vagas == 'Uruguai', 'Bairro' ] = 'Uruguai'
        df.loc[df.Vagas == 'São Cristóvão', 'Bairro' ] = 'São Cristóvão'
        df.loc[df.Vagas == 'Fátima', 'Bairro' ] = 'Fátima'
        df.loc[df.Vagas == 'Horto', 'Bairro' ] = 'Horto'
        df.loc[df.Vagas == 'Gurupi', 'Bairro' ] = 'Gurupi'
        df.loc[df.Vagas == 'Vale Quem Tem', 'Bairro' ] = 'Vale Quem Tem'
        df.loc[df.Vagas == 'São João', 'Bairro' ] = 'São João'
        df.loc[df.Vagas == 'Jóquei', 'Bairro' ] = 'Jóquei'
        df.loc[df.Vagas == 'Santa Isabel', 'Bairro' ] = 'Santa Isabel'
        df.loc[df.Vagas == 'Itararé', 'Bairro' ] = 'Itararé'
        df.loc[df.Vagas == 'Ilhotas', 'Bairro' ] = 'Ilhotas'
        df.loc[df.Vagas == 'Cristo Rei', 'Bairro' ] = 'Cristo Rei'
        df.loc[df.Vagas == 'Recanto das Palmeiras', 'Bairro' ] = 'Recanto das Palmeiras'
        df.loc[df.Vagas == 'Macaúba', 'Bairro' ] = 'Macaúba'
        df.loc[df.Vagas == 'Tabajaras', 'Bairro' ] = 'Tabajaras'
        df.loc[df.Vagas == 'Santa Maria', 'Bairro' ] = 'Santa Maria'
        df.loc[df.Vagas == 'Piçarra', 'Bairro' ] = 'Piçarra'
        df.loc[df.Vagas == 'Samapi', 'Bairro' ] = 'Samapi'
        df.loc[df.Vagas == 'Saci', 'Bairro' ] = 'Saci'
        df.loc[df.Vagas == 'Noivos', 'Bairro' ] = 'Noivos'
        df.loc[df.Vagas == 'Satélite', 'Bairro' ] = 'Satélite'
        df.loc[df.Vagas == 'Cabral', 'Bairro' ] = 'Cabral'
        df.loc[df.Vagas == 'Novo Uruguai', 'Bairro' ] = 'Novo Uruguai'
        df.loc[df.Vagas == 'Zoobotânico', 'Bairro' ] = 'Zoobotânico'
        df.loc[df.Vagas == 'Mocambinho', 'Bairro' ] = 'Mocambinho'
        df.loc[df.Vagas == 'Planalto', 'Bairro' ] = 'Planalto'
        df.loc[df.Vagas == 'Acarape', 'Bairro' ] = 'Acarape'
        df.loc[df.Vagas == 'Pirajá', 'Bairro' ] = 'Pirajá'
        df.loc[df.Vagas == 'Pedra Mole', 'Bairro' ] = 'Pedra Mole'
        df.loc[df.Vagas == 'Centro', 'Bairro' ] = 'Centro'
        df.loc[df.Vagas == 'Campestre', 'Bairro' ] = 'Campestre'
        df.loc[df.Vagas == 'Água Mineral', 'Bairro' ] = 'Água Mineral'
        df.loc[df.Vagas == 'Vale do Gavião', 'Bairro' ] = 'Vale do Gavião'
        df.loc[df.Vagas == 'Morada do Sol', 'Bairro' ] = 'Morada do Sol'
        df.loc[df.Vagas == 'Cidade Jardim', 'Bairro' ] = 'Cidade Jardim'
        df.loc[df.Vagas == 'Porto do Centro', 'Bairro' ] = 'Porto do Centro'
        df.loc[df.Vagas == 'Colorado', 'Bairro' ] = 'Colorado'
        df.loc[df.Vagas == 'Memorare', 'Bairro' ] = 'Memorare'
        df.loc[df.Vagas == 'São Pedro', 'Bairro' ] = 'São Pedro'
        df.loc[df.Vagas == 'Aeroporto', 'Bairro' ] = 'Aeroporto'

        # 5 ou mais 
        df.loc[df.Vagas == '5 ou mais', 'Vagas' ] = 5
        
        return df
    
    
    def data_transform( self, df):
        


        # Vagas as que estavam como o Bairro mudar para as mais comuns/moda
        df['Vagas'] = df.Vagas.apply(lambda x: x if type(x) == float else
                                             x if x == 5 else
                                       int(x) if x.isnumeric()  
                                       else df.Vagas.value_counts().index[0]             
                          )


        # Fillna Vagas - Mais COmun
        df.Vagas.fillna(df.Vagas.value_counts().index[0], inplace=True)

        # Quartos 5 ou mais  == 5
        df.loc[df.Quartos == '5 ou mais', 'Quartos' ] = 5
        df.loc[df.Banheiros == '5 ou mais', 'Banheiros' ] = 5

        # Quarto == 0 , moda
        df.loc[df.Quartos == 0, 'Quartos' ] = df.Quartos.value_counts().index[0]


        # Preço é nossa variável target - Fazer imputações nessa variável pode levar a viés 
        # Filtrar os Preços Nulos


        df = df[df.Preço.notnull() & df.Bairro.notnull()]


        # Removendo o m2 de Area Construída
        regex = '\d+'
        df['Area construida'] = df['Area construida'].apply(lambda x: re.match( regex, x ).group(0) if re.match( regex, x ) else x)

        # Area construida está area murada == 0
        #df.loc[df['Area construida'] == 'Área murada', 'Area construida'] = 0

        # Mudar dtype 
        df['Vagas'] = df.Vagas.astype(int)
        df['Banheiros'] = df.Banheiros.astype(int)
        df['Quartos'] = df.Quartos.astype(int)
        df['Area construida'] = df['Area construida'].astype(float)


        df['Preço'] = df['Preço'].astype(float)

        # convertendo os dados para categóricos
        for c in df.select_dtypes(include=['object']):
            df[c] = df[c].astype('category')

        return df


    def data_filter(self , df1):

        # Filtra valores preco e area construida com base na statistica
        df1 = df1[df1.Preço < 3000000][df1['Area construida'] < 800][df1.Preço > 100000]

        return df1
    

    def df1_merge_preco_metro(self, csv,df1 ):

        df_bairros_valor_metro = self.ler_csv(csv)
        df_bairros_valor_metro.drop('Unnamed: 0', axis=1, inplace=True)
        df_bairros_valor_metro.rename(columns={'preco_area':'preco_por_metro' }, inplace= True)
        df1 = pd.merge(df1, df_bairros_valor_metro, on= 'Bairro')

        return df1
    
    def feature_engineering( self, df1):

    
        df1['QuartosporAreaConstruida'] = (df1['Quartos'].astype(int)) / df1['Area construida']
        df1['LogAreadivBanheiro'] = np.log(df1['Banheiros'].astype(int) / df1['Area construida'])

        # Log das áreas
        df1['LogAreaConstruida'] = np.log1p(df1['Area construida'])

        categoria_e_bairro_por_area = df1.groupby(['Categoria', 'Bairro']).agg({'Area construida': 'mean'}).reset_index()

        df1['bairro_por_area'] = None

        for b in df1.Bairro.unique():
            for c in df1.Categoria.unique():
                df1.loc[(df1.Categoria == c) & (df1.Bairro == b), 'bairro_por_area' ] =round(categoria_e_bairro_por_area[categoria_e_bairro_por_area.Categoria == c][categoria_e_bairro_por_area.Bairro == b]['Area construida'].values[0], 2)


        df1.bairro_por_area = df1.bairro_por_area.astype('float')


        Leste = ['Jóquei', 'Jockey', 'Fátima', 'Horto', 'São Cristóvão', 'Ininga', 
        'Santa Isabel', 'Morada do Sol', 'Noivos', 'Morros', 'Campestre', 'Pedra Mole', 
        'Cidade Jardim', 'Novo Uruguai', 'Piçarreira', 'Planalto', 'Porto do Centro', 
        'Samapi', 'Santa Lia', 'Satélite', 'Socopó', 'Tabajaras', 'Uruguai', 
        'Vale do Gavião', 'Vale Quem Tem', 'Verde Lar', 'Árvores Verdes', 'São João', 
        'Zoobotânico', 'Recanto das Palmeiras']


        Sudeste = ['Beira Rio', 'Bom Princípio', 'Colorado', 'Comprida', 'Extrema', 
        'Flor do Campo', 'Gurupi', 'Itararé', 'Livramento', 'Novo Horizonte', 
        'Parque Ideal', 'Parque Poti', 'Redonda', 'Renascença', 'São Raimundo', 
        'São Sebastião', 'Tancredo Neves', 'Todos os Santos', 'Verde Cap']

        Sul = ['Angelim', 'Angélica', 'Areias', 'Bela Vista', 'Brasilar', 'Catarina', 
        'Cidade Nova', 'Cristo Rei', 'Distrito Industrial', 'Esplanada',
        'Lourival Parente', 'Macaúba', 'Monte Castelo', 'Morada Nova', 'Parque Jacinta', 
        'Parque Juliana', 'Parque Piauí', 'Parque São João', 'Parque Sul', 'Pedra Miúda', 
        'Pio XII', 'Portal Da Alegria', 'Promorar', 'Redenção', 'Saci', 'Santa Cruz',
        'Santa Luzia', 'Santo Antônio', 'São Lourenço', 'São Pedro', 'Tabuleta', 
        'Três Andares', 'Triunfo', 'Vermelha', 'Nossa Senhora Das Graças']

        Norte = ['Aroeiras', 'Acarape', 'Aeroporto', 'Água Mineral', 'Alegre'
        'Alto Alegre', 'Parque Alvorada', 'Bom Jesus', 'Buenos Aires', 'Cidade Industrial', 
        'Embrapa', 'Itaperu', 'Parque Brasil', 'Mafrense', 'Mafuá', 'Matadouro', 'Memorare', 
        'Monte Verde', 'Mocambinho', 'Morro da Esperança', 'Nova Brasília', 'Olarias'
        'Poti Velho', 'Primavera', 'Real Copagre', 'Santa Maria da Codipe', 'Santa Rosa', 
        'São Joaquim', 'Chapadinha', 'Jacinta Andrade', 'Pirajá', 'Vila São Francisco']

        Centro = ['Cabral', 'Centro Norte', 'Centro Sul', 'Porenquanto', 'Vila Operária', 
        'Matinha', 'Ilhotas', 'Frei Serafim', 'Marquês', 'Piçarra']




        df1["zona"] = df1.Bairro.apply(lambda x: "Leste" if x in Leste else 
                             "Norte" if x in Norte
                             else "Centro" if x in Centro else
                             "Sul" if x in Sul
                             else "Outros")

        return df1
    

    
    
    def data_preparation(self, df2):
    

        df2['QuartosporAreaConstruida'] = self.quarto_area_construida_scaler.fit_transform( df2[['QuartosporAreaConstruida']].values )

        df2['preco_por_metro'] = self.preco_por_metro_scaler.fit_transform( df2[['preco_por_metro']].values )


        df2['bairro_por_area'] = self.bairro_por_area_scaler.fit_transform( df2[['bairro_por_area']].values )


        #  - Label Encoding

        df2['Bairro'] =self.bairro_scaler.fit_transform( df2['Bairro'] )


        df2['Banheiros'] = self.banheiros_scaler.fit_transform(df2.Banheiros.astype(int))   

        df2['Vagas'] = self.vagas_scaler.fit_transform(df2.Vagas.astype(int))  
        
        return df2
    
    def get_prediction( self, test_data ):
        # prediction
        pred = self.model.predict( test_data )
        
        # join pred into the original data
        predict = np.expm1( pred )
        
        return predict


