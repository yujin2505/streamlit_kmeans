import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer  #컬럼 트랜스포머 대신에 더미스를 사용해도 된다.

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title('K-Means 클러스터링 앱')
    st.text('csv 파일을 업로드하면, 비슷한 유형의 데이터끼리 묶어주는 앱입니다.')
    
    # 1. csv 파일 업로드
    
    file = st.file_uploader('CSV파일 업로드', type=['csv'])
        
    if file is not None :
        
        # 1-1. 판다스의 데이터프레임으로 읽는다
        
        df = pd.read_csv(file)
        
        # 1-2. 10개 미만의 파일을 올리면, 에러처리하자
        
        print(df.shape)
        
        if df.shape[0] < 10 :
            st.error('데이터의 갯수는 10개 이상이어야 합니다')
            return
        
        # 1-3. 유저한테 데이터프레임 보여준다
        
        st.dataframe(df)
        
        # 2. nan 데이터 있으면, 삭제하자.
        
        print( df.isna().sum() )
        
        st.subheader('각 컬럼별 Nan의 갯수입니다.')
        
        st.dataframe(df.isna().sum())
        
        df.dropna(inplace=True)
        
        df.reset_index(inplace=True, drop=True) 
        
        st.info('Nan이 있으면 해당 데이터는 삭제합니다')
        

        # 3. 유저한테 컬럼을 선택할 수 있도록 하자
        
        print( df.columns )
        
        st.subheader('클러스터링에 사용할 컬럼 선택!')
        
        selected_columns = st.multiselect('X로 사용할 컬럼을 선택하세요', df.columns)
        
        X = df[selected_columns]
        
        st.dataframe(X)
        
        if len(selected_columns) >= 2 :
            
            X_new = pd.DataFrame()  # 빈 데이터프레임 만들기 
            
            print(X_new)
            
            # 4. 해당 컬럼의 데이터가 문자열이면, 숫자로 바꿔주자.
            
            for column in X.columns :
                print( X[column].dtype ) 
                
                # 컬럼의 데이터가 문자열이면, 레이블인코딩 또는 원핫인코딩 해야 한다
                if X[column].dtype == object :
                    
                    if X[column].nunique() >= 3 :          
                        column_names = sorted( X[column].unique() )   #원핫 인코딩
                        X_new[column_names] = pd.get_dummies( X[column].to_frame() )  #비어있는 데이터프레임에 컬럼 추가
                    
                    else :
                        encoder = LabelEncoder() #레이블 인코딩
                        X_new[column] = encoder.fit_transform( X[column] )                    
                    
                else :  
                    X_new[column] = X[column] #숫자데이터처리
            
            
            # 4-1 인덱스값을 다시 초기화시켜준다
           
            X_new.reset_index(inplace=True, drop=True)
            
            
            # 4-2 X_new는 숫자로만 되어있는 데이터프레임이다 유저한테 보여주자
            
            st.subheader('클러스터링에 실제 사용할 데이터')
            st.dataframe(X_new)
            
            # 5. k의 갯수를 1개부터 10개까지 해서 wcss를 구한다
            
            # 6. elbow method를 이용해서, 차트로 보여준다
            
            # 7. 유저가 k의 갯수를 지정한다
            
            # 8. K-Means 수행해서 그룹정보를 가져온다
            
            # 9. 원래 있던 df에 Group이라는 새로운 column을 만들어 데이터를 넣어준다.     
            
            # 10. 결과를 파일로 저장한다.
    

if __name__ == '__main__' :
    main()