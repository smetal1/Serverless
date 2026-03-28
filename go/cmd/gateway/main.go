// Package main is the entrypoint for the Podstack API gateway.
// It serves OpenAI-compatible API endpoints and routes requests to model pods.
package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	v1 "github.com/podstack/serverless/api/v1"
	"github.com/podstack/serverless/internal/gateway"
)

var (
	scheme = runtime.NewScheme()
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1.AddToScheme(scheme))
}

func main() {
	var (
		addr      string
		namespace string
	)

	flag.StringVar(&addr, "addr", ":8080", "The address the gateway listens on.")
	flag.StringVar(&namespace, "namespace", "podstack-system", "The namespace to watch for model deployments.")
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))
	log := ctrl.Log.WithName("gateway")

	// Create K8s client
	cfg := ctrl.GetConfigOrDie()
	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		log.Error(err, "unable to create K8s client")
		os.Exit(1)
	}

	// Create and start server
	srv := gateway.NewServer(gateway.Config{
		Addr:         addr,
		K8sClient:    k8sClient,
		Namespace:    namespace,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
	}, log)

	// Graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		log.Info("shutdown signal received")
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutdownCancel()
		if err := srv.Stop(shutdownCtx); err != nil {
			log.Error(err, "server shutdown error")
		}
		cancel()
	}()

	log.Info("starting Podstack gateway", "addr", addr, "namespace", namespace)
	if err := srv.Start(ctx); err != nil {
		log.Error(err, "server error")
		os.Exit(1)
	}
}
