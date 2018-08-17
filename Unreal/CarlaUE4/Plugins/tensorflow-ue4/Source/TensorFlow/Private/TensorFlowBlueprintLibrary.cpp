// Fill out your copyright notice in the Description page of Project Settings.

#include "TensorFlowPrivatePCH.h"
#include "RHI.h"
#include "Engine.h"
#include "Runtime/CoreUObject/Public/UObject/UObjectGlobals.h"
#include "TensorFlowBlueprintLibrary.h"


TArray<float> UTensorFlowBlueprintLibrary::Conv_GreyScaleTexture2DToFloatArray(UTexture2D* InTexture)
{
	TArray<float> FloatArray;
	FloatArray.SetNum(InTexture->GetSizeX()* InTexture->GetSizeY());

	// Lock the texture so it can be read
	uint8* MipData = static_cast<uint8*>(InTexture->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_ONLY));

	for (int i = 0; i < FloatArray.Num(); i++)
	{
		int MipPointer = i * 4;
		float GreyscaleValue = (MipData[MipPointer] + MipData[MipPointer + 1] + MipData[MipPointer + 2]) / 3.f;
		FloatArray[i] = GreyscaleValue / 255.f;	 //normalize it
	}

	// Unlock the texture
	InTexture->PlatformData->Mips[0].BulkData.Unlock();

	return FloatArray;
}

TArray<float> UTensorFlowBlueprintLibrary::Conv_Texture2DToFloatArray(UTexture2D* InTexture)
{
	TArray<float> FloatArray;

	//RGBA texture
	FloatArray.SetNum(InTexture->GetSizeX()* InTexture->GetSizeY() * 4);

	//Lock the texture so it can be read
	uint8* MipData = static_cast<uint8*>(InTexture->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_ONLY));

	for (int i = 0; i < FloatArray.Num(); i++)
	{
		int MipPointer = i * 4;
		FloatArray[i] = MipData[i] / 255.f; //normalize value to 0-1.0f
	}

	// Unlock the texture
	InTexture->PlatformData->Mips[0].BulkData.Unlock();

	return FloatArray;
}

TArray<float> UTensorFlowBlueprintLibrary::InvertFloatArray(const TArray<float>& InFloatArray)
{
	TArray<float> InvertedArray;
	InvertedArray.SetNum(InFloatArray.Num());

	for (int i=0;i<InFloatArray.Num();i++)
	{
		InvertedArray[i] = 1 - InFloatArray[i];
	}

	return InvertedArray;
}


UTexture2D* UTensorFlowBlueprintLibrary::Conv_GrayScaleFloatArrayToTexture2D(const TArray<float>& InFloatArray, const FVector2D InSize /*= FVector2D(0,0)*/)
{
	FVector2D Size;

	//Create square image and lock for writing
	if (InSize == FVector2D(0, 0))
	{
		int32 Length = FMath::Pow(InFloatArray.Num(), 0.5);
		if (Length * Length != InFloatArray.Num())
		{
			UE_LOG(TensorFlowLog, Warning, TEXT("Invalid float array without specified size, needs to be square."));
			return nullptr;
		}
		Size = FVector2D(Length, Length);
	}
	else
	{
		Size = InSize;
	}

	UTexture2D* Pointer = UTexture2D::CreateTransient(Size.X, Size.Y, PF_R8G8B8A8);
	uint8* MipData = static_cast<uint8*>(Pointer->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_WRITE));

	//Copy Data
	for (int i = 0; i < InFloatArray.Num(); i++)
	{
		int MipPointer = i * 4;
		int GreyValue = InFloatArray[i] * 255.f;
		MipData[MipPointer] = GreyValue;
		MipData[MipPointer + 1] = GreyValue;
		MipData[MipPointer + 2] = GreyValue;
		MipData[MipPointer + 3] = 255;	//Alpha
	}

	//Unlock and Return data
	Pointer->PlatformData->Mips[0].BulkData.Unlock();
	Pointer->UpdateResource();
	return Pointer;
}

UTexture2D* UTensorFlowBlueprintLibrary::Conv_FloatArrayToTexture2D(const TArray<float>& InFloatArray, const FVector2D InSize /*= FVector2D(0, 0)*/)
{
	FVector2D Size;

	//Create square image and lock for writing
	if (InSize == FVector2D(0, 0))
	{
		int32 Length = FMath::Pow(InFloatArray.Num(), 0.5);
		if (Length * Length != InFloatArray.Num())
		{
			UE_LOG(TensorFlowLog, Warning, TEXT("Invalid float array without specified size, needs to be square."));
			return nullptr;
		}
		Size = FVector2D(Length, Length);
	}
	else
	{
		Size = InSize;
	}

	UTexture2D* Pointer = UTexture2D::CreateTransient(Size.X, Size.Y, PF_R8G8B8A8);
	uint8* MipData = static_cast<uint8*>(Pointer->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_WRITE));

	//Copy Data
	for (int i = 0; i < InFloatArray.Num(); i++)
	{
		MipData[i] = int(InFloatArray[i] * 255.f);
	}

	//Unlock and Return data
	Pointer->PlatformData->Mips[0].BulkData.Unlock();
	Pointer->UpdateResource();
	return Pointer;
}

UTexture2D* UTensorFlowBlueprintLibrary::Conv_RenderTargetTextureToTexture2D(UTextureRenderTarget2D* PassedTexture)
{
	int TextureLength = PassedTexture->SizeX * PassedTexture->SizeY;
	UTexture2D* Pointer = UTexture2D::CreateTransient(PassedTexture->SizeX, PassedTexture->SizeY, PF_R8G8B8A8);
	
	TArray<FColor> SurfData;
	FRenderTarget *RenderTarget = PassedTexture->GameThread_GetRenderTargetResource();
	RenderTarget->ReadPixels(SurfData);

	uint8* MipData = static_cast<uint8*>(Pointer->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_WRITE));

	//Copy Data
	for (int i = 0; i < TextureLength; i++)
	{
		int MipPointer = i * 4;
		const FColor& Color = SurfData[i];
		uint8 AdjustedColor = (Color.R + Color.G + Color.B) / 3;
		MipData[MipPointer] = AdjustedColor;
		MipData[MipPointer + 1] = AdjustedColor;
		MipData[MipPointer + 2] = AdjustedColor;
		MipData[MipPointer + 3] = 255;	//Alpha
	}

	//Unlock and Return data
	Pointer->PlatformData->Mips[0].BulkData.Unlock();
	Pointer->UpdateResource();
	PassedTexture->Source.UnlockMip(0);
	return Pointer;
}

TArray<float> UTensorFlowBlueprintLibrary::Conv_ByteToFloatArray(const TArray<uint8>& InByteArray, float Scale)
{
	TArray<float> FloatArray;
	FloatArray.SetNum(InByteArray.Num());

	int32 Size = InByteArray.Num();
	for (int i = 0; i< Size; i++)
	{
		FloatArray[i] = float(InByteArray[i]) * Scale;
	}

	return FloatArray;
}

TArray<float> UTensorFlowBlueprintLibrary::Conv_TextureRender2DToFloatArray(UTextureRenderTarget2D* CaptureRenderTarget, bool bGrey)
{
	TArray<FColor> SurfData;
	FRenderTarget *RenderTarget = CaptureRenderTarget->GameThread_GetRenderTargetResource();
	RenderTarget->ReadPixels(SurfData);

	TArray<float> FloatArray;

	size_t nListNum = CaptureRenderTarget->SizeX* CaptureRenderTarget->SizeY;

	if (bGrey)
	{
		FloatArray.SetNum(nListNum);

		for (int i = 0; i < nListNum; i++)
		{
			const FColor& Color = SurfData[i];

			float GreyscaleValue = (Color.R + Color.G + Color.B) / 3.f;
			FloatArray[i] = GreyscaleValue;	 //normalize it
		}
	}
	else
	{
		int nRGBAScale = 4;
		FloatArray.SetNum(nListNum * nRGBAScale);

		for (int i = 0; i < nListNum; i++)
		{
			const FColor& Color = SurfData[i];
			FloatArray[(i * nRGBAScale)] = Color.R / 255.f;
			FloatArray[(i * nRGBAScale) + 1] = Color.G / 255.f;
			FloatArray[(i * nRGBAScale) + 2] = Color.B / 255.f;
			FloatArray[(i * nRGBAScale) + 3] = Color.A / 255.f;
		}
	}
	CaptureRenderTarget->Source.UnlockMip(0);
	return FloatArray;
}

UTextureRenderTarget2D* UTensorFlowBlueprintLibrary::Conv_FloatArraytoTextureRender2D(const TArray<float>& InFloatArray, int SizeX, int SizeY)
{
	UTextureRenderTarget2D* TextureTarget = NewObject<UTextureRenderTarget2D>();
	TextureTarget->InitAutoFormat(SizeX, SizeY);

	TArray<FColor> SurfData;

	FRenderTarget *RenderTarget = TextureTarget->GameThread_GetRenderTargetResource();
	RenderTarget->ReadPixels(SurfData);

	size_t nListNum = TextureTarget->SizeX* TextureTarget->SizeY;

	int nRGBAScale = 4;

	for (int i = 0; i < nListNum; i++)
	{
		SurfData[i].R = InFloatArray[(i * nRGBAScale)] * 255.f;
		SurfData[i].G = InFloatArray[(i * nRGBAScale) + 1] * 255.f;
		SurfData[i].B = InFloatArray[(i * nRGBAScale) + 2] * 255.f;
		SurfData[i].A = InFloatArray[(i * nRGBAScale) + 3] * 255.f;
	}

	TextureTarget->Source.UnlockMip(0);
	return TextureTarget;
}
