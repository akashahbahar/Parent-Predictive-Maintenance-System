# Use the official .NET 8 SDK image for build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy csproj files and restore as distinct layers
COPY ["PredictiveMaintenance.API/PredictiveMaintenance.API.csproj", "PredictiveMaintenance.API/"]
COPY ["PredictiveMaintenanceSystem/PredictiveMaintenanceSystem.csproj", "PredictiveMaintenanceSystem/"]
RUN dotnet restore "PredictiveMaintenance.API/PredictiveMaintenance.API.csproj"

# Copy the rest of the source code
COPY PredictiveMaintenance.API/. ./PredictiveMaintenance.API/
COPY PredictiveMaintenanceSystem/. ./PredictiveMaintenanceSystem/

WORKDIR /src/PredictiveMaintenance.API
RUN dotnet publish -c Release -o /app/publish

# Build runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app

# Expose port 80 in the container
EXPOSE 80

# Copy published app
COPY --from=build /app/publish .

# If your model.zip is generated at build time and needed at runtime, ensure it's copied:
# (Uncomment if needed, or ensure your .csproj copies it to output)
# COPY PredictiveMaintenanceSystem/model.zip ./model.zip

ENTRYPOINT ["dotnet", "PredictiveMaintenance.API.dll"]