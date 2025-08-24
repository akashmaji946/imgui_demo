#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef OFF_READER_H
#define OFF_READER_H

typedef struct Vt {
	float x,y,z;
}Vertex;

typedef struct Pgn {
	int noSides;
	int *v;
}Polygon;

typedef struct offmodel {
	Vertex *vertices;
	Polygon *polygons;
	int numberOfVertices;
 	int numberOfPolygons;
}OFFModel;

OFFModel* readOffFile(char * OffFile) {
	printf(">> Reading OFF file: %s\n", OffFile);
	FILE * input;
	char type[4] = {0};
	int noEdges;
	int i,j;
	float x,y,z;
	int n, v;
	int nv, np;
	OFFModel *model;
	printf(">> Opening file: %s\n", OffFile);
	input = fopen(OffFile, "r");
	if(input == NULL) {
		printf("Failed to open file: %s\n", OffFile);
		exit(1);
	}
	fscanf(input, "%s\n", type);
	/* First line should be OFF */
	printf(">> Type: %s", type);
	if(strcmp(type,"OFF") != 0) {
		printf("Not a OFF file\n");
		exit(1);
	}
	/* Read the no. of vertices, faces and edges */
	fscanf(input, "%d", &nv);
	fscanf(input, "%d", &np);
	fscanf(input, "%d", &noEdges);

	model = (OFFModel*)malloc(sizeof(OFFModel));
	model->numberOfVertices = nv;
	model->numberOfPolygons = np;
	
	
	/* allocate required data */
	model->vertices = (Vertex*) malloc(nv * sizeof(Vertex));
	model->polygons = (Polygon*) malloc(np * sizeof(Polygon));
	

	/* Read the vertices' location*/	
	for(i = 0;i < nv;i ++) {
		fscanf(input, "%f %f %f", &x,&y,&z);
		(model->vertices[i]).x = x;
		(model->vertices[i]).y = y;
		(model->vertices[i]).z = z;
	}

	/* Read the Polygons */	
	for(i = 0;i < np;i ++) {
		/* No. of sides of the polygon (Eg. 3 => a triangle) */
		fscanf(input, "%d", &n);
		
		(model->polygons[i]).noSides = n;
		(model->polygons[i]).v = (int *) malloc(n * sizeof(int));
		/* read the vertices that make up the polygon */
		for(j = 0;j < n;j ++) {
			fscanf(input, "%d", &v);
			(model->polygons[i]).v[j] = v;
		}
	}

	fclose(input);
	return model;
}

void printOffModel(OFFModel* model) {
	// int i, j;
	//printf("OFF\n");
	// printf("%d %d 0 \n", model->numberOfVertices, model->numberOfPolygons);
	
	// for(i = 0; i < model->numberOfVertices;i ++) {
	// 	printf("%f %f %f \n", (model->vertices[i]).x, (model->vertices[i]).y, (model->vertices[i]).z);
	// }
	// for(i = 0;i < model->numberOfPolygons;i ++) {
	// 	printf("%d ", (model->polygons[i]).noSides);
	// 	for(j = 0;j < (model->polygons[i]).noSides;j ++) {
	// 		printf("%d ", (model->polygons[i]).v[j]);
	// 	}
	// 	printf("\n");
	// }
	
}

int FreeOffModel(OFFModel* model) {
	int i,j;
	if(model == NULL){
		return 0;
	}
	free(model->vertices);
	for(i = 0; i < model->numberOfPolygons; ++i ){
		if((model->polygons[i]).v){
			free((model->polygons[i]).v);
		}
	}
	free(model->polygons);
	free(model);
	return 1;
}

// int main(int argc, char ** argv) {
// 	OFFModel *model = readOffFile(argv[1]);
// 	printOffModel(model);
// 	FreeOffModel(model);
// 	return 0;
// }


#endif // OFF_READER_H

