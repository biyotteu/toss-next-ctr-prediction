#!/usr/bin/env python3
"""
train.parquet에서 500개의 샘플을 추출하여 test_train.parquet로 저장하는 스크립트
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def sample_data():
    """원본 데이터에서 500개 샘플을 추출하여 저장"""
    
    # 파일 경로 설정
    input_file = "./data/train.parquet"
    output_file = "./data/test_train.parquet"
    
    # 입력 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return False
    
    try:
        print(f"📂 원본 데이터 읽는 중: {input_file}")
        
        # 원본 데이터의 크기를 먼저 확인
        df_info = pd.read_parquet(input_file, engine='pyarrow')
        total_rows = len(df_info)
        print(f"📊 전체 데이터 행 수: {total_rows:,}")
        
        # 500개보다 적은 경우 처리
        if total_rows <= 2000:
            print(f"⚠️  전체 데이터가 500개 이하입니다. 모든 데이터({total_rows}개)를 사용합니다.")
            sample_size = total_rows
            sampled_df = df_info
        else:
            print(f"🎯 {total_rows:,}개 중 500개를 랜덤 샘플링합니다...")
            
            # 메모리 효율성을 위해 random seed 설정
            np.random.seed(42)
            
            # 랜덤 인덱스 생성
            sample_indices = np.random.choice(total_rows, size=5000, replace=False)
            sample_indices = sorted(sample_indices)  # 정렬하여 IO 효율성 향상
            
            # 샘플 데이터 추출
            sampled_df = df_info.iloc[sample_indices].copy()
            sample_size = 5000
        
        # 출력 디렉토리 생성 (필요한 경우)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 샘플 데이터 저장 중: {output_file}")
        
        # 샘플 데이터를 parquet 형식으로 저장
        sampled_df.to_parquet(
            output_file, 
            engine='pyarrow',
            compression='snappy',  # 압축으로 파일 크기 최적화
            index=False
        )
        
        # 결과 확인
        saved_df = pd.read_parquet(output_file)
        saved_rows = len(saved_df)
        
        print(f"✅ 성공적으로 완료!")
        print(f"   - 샘플링된 데이터: {sample_size:,}개")
        print(f"   - 저장된 데이터: {saved_rows:,}개")
        print(f"   - 저장 위치: {output_file}")
        
        # 데이터 미리보기
        print(f"\n📋 샘플 데이터 미리보기:")
        print(f"   - 컬럼 수: {len(saved_df.columns)}")
        print(f"   - 컬럼 이름: {list(saved_df.columns[:5])}{'...' if len(saved_df.columns) > 5 else ''}")
        print(f"   - 데이터 타입:")
        for col in saved_df.columns[:3]:  # 처음 3개 컬럼만 표시
            print(f"     {col}: {saved_df[col].dtype}")
        if len(saved_df.columns) > 3:
            print(f"     ... (총 {len(saved_df.columns)}개 컬럼)")
            
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 데이터 샘플링 시작...")
    success = sample_data()
    
    if success:
        print("\n🎉 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n💥 작업 중 오류가 발생했습니다.")
        exit(1)
